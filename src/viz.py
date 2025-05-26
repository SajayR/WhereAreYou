
import textwrap
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple


_DEF_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
_DEF_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def denorm(img: torch.Tensor) -> np.ndarray:
    np_img = img.permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * _DEF_STD + _DEF_MEAN) * 255
    return np.clip(np_img, 0, 255).astype(np.uint8)


class TextVisualizer:
    def __init__(self, patch_size: int = 14, image_size: int = 224):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = image_size // patch_size
        colors = [
            (0, 0, 0, 0),
            (0, 0, 1, 0.5),
            (1, 0, 0, 0.7),
            (1, 1, 0, 1.0),
        ]
        self.cmap = LinearSegmentedColormap.from_list("custom", colors)

    def _patches_to_heatmaps(self, patch_att: torch.Tensor) -> torch.Tensor:
        n, _ = patch_att.shape
        patches = patch_att.reshape(n, self.num_patches, self.num_patches)
        patches = patches ** 2
        return F.interpolate(
            patches.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    @staticmethod
    def _group_tokens(tokens: List[str], offsets: List[Tuple[int, int]], text: str):
        words, groups = [], []
        cur_word, cur_group = "", []
        for idx, (tok, (s, e)) in enumerate(zip(tokens, offsets)):
            if s == 0 or text[s - 1].isspace():
                if cur_group:
                    words.append(cur_word)
                    groups.append(cur_group)
                cur_word, cur_group = text[s:e], [idx]
            else:
                cur_word += text[s:e]
                cur_group.append(idx)
        if cur_group:
            words.append(cur_word)
            groups.append(cur_group)
        return words, groups

    def _merge(self, maps: torch.Tensor, groups: List[List[int]]):
        return torch.stack([maps[g].mean(0) if len(g) > 1 else maps[g[0]] for g in groups])

    def _overlay(self, img: np.ndarray, heat: np.ndarray, alpha: float = 0.3):
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat = np.power(heat, 2)
        heat_rgb = (self.cmap(heat)[..., :3] * 255).astype(np.uint8)
        return ((1 - alpha) * img + alpha * heat_rgb).astype(np.uint8)

    @staticmethod
    def _draw_caption(ax, full: str, word: str,
                      max_chars: int = 45, font_size: int = 14):
        """
        Draw `full` caption under an axis, highlighting one `word` in red.

        Parameters
        ----------
        ax        : matplotlib axis – target to draw into (will be cleared).
        full      : full caption string.
        word      : word to highlight (case‑insensitive, whole‑word match).
        max_chars : soft wrap width in characters.
        font_size : point size for the caption font.
        """
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        import re, textwrap
        for fname in ["Arial", "DejaVuSans", "FreeSans", "LiberationSans"]:
            try:
                font = ImageFont.truetype(fname, font_size)
                break
            except Exception:
                font = None
        if font is None:
            font = ImageFont.load_default()

        lines = textwrap.wrap(full, max_chars)

        def text_dims(txt: str):
            try:
                w, h = font.getbbox(txt)[2:]
            except AttributeError:
                w, h = font.getsize(txt)
            return w, h

        line_h = max(text_dims(l)[1] for l in lines) + 4
        img_w  = max(text_dims(l)[0] for l in lines) + 20
        img_h  = line_h * len(lines)

        img = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        pattern = r"\b" + re.escape(word) + r"\b"
        y = 0
        for line in lines:
            x = img_w / 2 - text_dims(line)[0] / 2
            cur = 0
            for m in re.finditer(pattern, line, re.IGNORECASE):
                s, e = m.span()
                if s > cur:
                    seg = line[cur:s]
                    draw.text((x, y), seg, fill=(0, 0, 0), font=font)
                    x += text_dims(seg)[0]
                seg = line[s:e]
                draw.text((x, y), seg, fill=(255, 0, 0), font=font)
                x += text_dims(seg)[0]
                cur = e
            if cur < len(line):
                seg = line[cur:]
                draw.text((x, y), seg, fill=(0, 0, 0), font=font)

            y += line_h

        ax.imshow(np.asarray(img))
        ax.axis("off")

    def plot_token_attentions(self, model, frame, text,
                              output_path: str | None = None,
                              dpi: int = 300):

        CAP_FONT_SIZE = 18  # pt
        CAP_WRAP      = 45   # chars/line
        IMG_RATIO     = 5.0  # relative height of an image row
        LINE_RATIO    = 0.28 # reslative height contribution per caption line

        model.eval()
        with torch.no_grad():
            #frame = frame.unsqueeze(0)
            #frame = frame.to(dtype=torch.bfloat16)
            vf = model.visual_embedder(frame.unsqueeze(0))#.to(dtype=torch.bfloat16))
            vf = vf.to(dtype=torch.float32)
            tf, mask = model.text_embedder([text])
            mask = mask.to(dtype=torch.float32)
            tf = tf.to(dtype=torch.float32)
            sims = model.compute_similarity_matrix(tf, vf).squeeze(0)[: int(mask.sum())]

        tokenizer = model.text_embedder.tokenizer
        enc   = tokenizer(text, add_special_tokens=False,
                          return_offsets_mapping=True)
        toks  = [t.replace("Ġ", "").replace("▁", "")
                 for t in tokenizer.convert_ids_to_tokens(enc["input_ids"])]
        words, groups = self._group_tokens(toks, enc["offset_mapping"], text)

        word_maps = self._merge(self._patches_to_heatmaps(sims), groups)
        img_np    = denorm(frame)

        n_cells = len(words) + 1           
        cols    = min(4, n_cells)
        rows    = (n_cells + cols - 1) // cols

        import textwrap
        line_counts = [len(textwrap.wrap(text, CAP_WRAP))]      # GT caption
        line_counts += [len(textwrap.wrap(text, CAP_WRAP))
                        for _ in words]                         # every word
        max_lines   = max(line_counts)
        cap_ratio   = max_lines * LINE_RATIO
        from matplotlib.gridspec import GridSpec
        fig_h = (IMG_RATIO + cap_ratio) * rows * 1.1
        fig   = plt.figure(figsize=(4.8 * cols, fig_h),
                           constrained_layout=False)

        gs = GridSpec(rows * 2, cols,
                      height_ratios=sum(([IMG_RATIO, cap_ratio] for _ in range(rows)), []),
                      hspace=0.25, wspace=0.02, figure=fig)

        def panel(r, c):
            return fig.add_subplot(gs[r, c])
        ax_img = panel(0, 0)
        ax_img.imshow(img_np);  ax_img.axis("off")
        ax_img.set_title("Ground truth", pad=4)
        panel(1, 0).axis("off")  

        for k, (w, hmap) in enumerate(zip(words, word_maps), start=1):
            r, c = divmod(k, cols)
            ax_img = panel(2 * r,     c)
            ax_cap = panel(2 * r + 1, c)

            ax_img.imshow(self._overlay(img_np, hmap.cpu().numpy()))
            ax_img.axis("off")

            ax_cap.axis("off")
            self._draw_caption(ax_cap, text, w,
                               max_chars=CAP_WRAP,
                               font_size=CAP_FONT_SIZE)

        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
