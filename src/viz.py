# viz.py (refined)

"""
Improved text‑visual attention visualiser with:
* ground‑truth frame first
* word‑level averaged heat‑maps
* caption under each cell, attended word in **red**
* layout fixes so captions are always visible

Drop‑in replacement – same public API.
"""
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
import math 
###############################################################################
# Helper utilities
###############################################################################

_DEF_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
_DEF_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def denorm(img: torch.Tensor) -> np.ndarray:
    """ImageNet‑denormalise and to uint8 HWC."""
    np_img = img.permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * _DEF_STD + _DEF_MEAN) * 255
    return np.clip(np_img, 0, 255).astype(np.uint8)

###############################################################################
# Text visualiser
###############################################################################

class TextVisualizer:
    def __init__(self, patch_size: int = 14, image_size: int = 224, max_heads_vis: int = 4):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = image_size // patch_size
        self.max_heads_vis = max_heads_vis 
        colors = [
            (0, 0, 0, 0),
            (0, 0, 1, 0.5),
            (1, 0, 0, 0.7),
            (1, 1, 0, 1.0),
        ]
        self.cmap = LinearSegmentedColormap.from_list("custom", colors)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
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

    # --------------------------------------------------------------------- #
    # Caption drawing – split into 3 text objects so we can colour the word
    # --------------------------------------------------------------------- #
    ###############################################################################
# Caption drawing helper (signature extended)
###############################################################################
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

        # pick a widely‑available sans‑serif font or fall back
        for fname in ["Arial", "DejaVuSans", "FreeSans", "LiberationSans"]:
            try:
                font = ImageFont.truetype(fname, font_size)
                break
            except Exception:
                font = None
        if font is None:
            font = ImageFont.load_default()

        lines = textwrap.wrap(full, max_chars)

        # helper to measure text reliably across Pillow versions
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

                # text before the match
                if s > cur:
                    seg = line[cur:s]
                    draw.text((x, y), seg, fill=(0, 0, 0), font=font)
                    x += text_dims(seg)[0]

                # the highlighted match
                seg = line[s:e]
                draw.text((x, y), seg, fill=(255, 0, 0), font=font)
                x += text_dims(seg)[0]
                cur = e

            # remainder of the line
            if cur < len(line):
                seg = line[cur:]
                draw.text((x, y), seg, fill=(0, 0, 0), font=font)

            y += line_h

        ax.imshow(np.asarray(img))
        ax.axis("off")

###############################################################################
# Public API – updated to enforce uniform cell size
###############################################################################
    def plot_token_attentions(self, model, frame, text,
                              output_path: str | None = None,
                              dpi: int = 300):
        # ------------------------------------------------------------ #
        # Tunable layout constants (change here to tweak look‑and‑feel)
        # ------------------------------------------------------------ #
        CAP_FONT_SIZE = 18  # pt
        CAP_WRAP      = 45   # chars/line
        IMG_RATIO     = 5.0  # relative height of an image row
        LINE_RATIO    = 0.28 # relative height contribution per caption line
        # ------------------------------------------------------------ #

        model.eval()
        with torch.no_grad():
            # Assuming text_embedder returns (feats, mask) tuple
            vf, (tf, mask) = (
                model.visual_embedder(frame.unsqueeze(0)),
                model.text_embedder([text])
            )

            H   = getattr(model, 'num_heads', 1)
            H   = max(1, H)                           # safety
            # Assuming self.max_heads_vis exists or default to H
            Hvis = min(H, getattr(self, 'max_heads_vis', H)) # cap for readability

            B, Nt, D = tf.shape # B is 1 here
            Nv       = vf.shape[1]
            Dh       = D // H
            # Reshape features per head, removing batch dim (B=1)
            t = tf.squeeze(0).view(Nt, H, Dh).permute(1, 0, 2)          # H, Nt, Dh
            v = vf.squeeze(0).view(Nv, H, Dh).permute(1, 0, 2)          # H, Nv, Dh
            # Compute similarity per head
            sims_head = torch.matmul(t, v.transpose(-2, -1)) * model.temperature  # H,Nt,Nv
            # Trim padding based on mask and limit heads to Hvis
            sims_head = sims_head[:Hvis, : int(mask.sum())]   # Hvis, Nt_actual, Nv

        tokenizer = model.text_embedder.tokenizer
        enc   = tokenizer(text, add_special_tokens=False,
                          return_offsets_mapping=True)
        toks  = [t.replace("Ġ", "").replace(" ", "")
                 for t in tokenizer.convert_ids_to_tokens(enc["input_ids"])]
        words, groups = self._group_tokens(toks, enc["offset_mapping"], text)

        # ----- produce heat-maps for every head we visualise -----
        head_word_maps = [
            self._merge(self._patches_to_heatmaps(sims_head[h]), groups)
            for h in range(Hvis)                                                # list length Hvis
        ]
        img_np    = denorm(frame) # Assuming denorm is defined elsewhere

        n_cells = len(words) + 1           # ground truth + every word
        cols    = min(4, n_cells)
        rows    = (n_cells + cols - 1) // cols * (Hvis)   # one strip per head

        # ---------- determine uniform caption height ---------------- #
        import textwrap
        line_counts = [len(textwrap.wrap(text, CAP_WRAP))]      # GT caption
        line_counts += [len(textwrap.wrap(text, CAP_WRAP))
                        for _ in words]                         # every word
        max_lines   = max(line_counts)
        cap_ratio   = max_lines * LINE_RATIO

        # ---------- build the figure/grid --------------------------- #
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt # Ensure plt is imported
        fig_h = (IMG_RATIO + cap_ratio) * rows * 1.1 # Use NEW rows
        fig   = plt.figure(figsize=(4.8 * cols, fig_h),
                           constrained_layout=False)

        gs = GridSpec(rows * 2, cols, # Use NEW rows
                      height_ratios=sum(([IMG_RATIO, cap_ratio]
                                         for _ in range(rows)), []), # Use NEW rows
                      hspace=0.25, wspace=0.02, figure=fig)

        def panel(r, c):
            # Ensure r is within bounds for gs
            if r < rows * 2:
                 return fig.add_subplot(gs[r, c])
            else:
                 # Handle potential index out of bounds, though logic should prevent this
                 print(f"Warning: Attempting to access invalid grid row {r}")
                 return fig.add_subplot(gs[0,0]) # Fallback or raise error

        panel_idx = 0

        # loop over heads first, then words
        for h in range(Hvis):
            # ----- ground truth for this head row -----
            if h > 0:
                panel_idx = math.ceil(panel_idx / cols) * cols
            r, c = divmod(panel_idx, cols)
            ax_img = panel(2 * r, c)
            ax_img.imshow(img_np); ax_img.axis("off")
            ax_img.set_title(f"GT  (head {h})", pad=4)
            panel(2 * r + 1, c).axis("off") # Blank caption panel
            panel_idx += 1

            # ----- every word for this head -----
            # Ensure head_word_maps[h] has the same length as words
            if len(head_word_maps[h]) != len(words):
                 print(f"Warning: Mismatch between words ({len(words)}) and heatmaps ({len(head_word_maps[h])}) for head {h}")
                 continue # Skip this head or handle error appropriately

            for w, hmap in zip(words, head_word_maps[h]):
                if panel_idx >= rows * cols: # Safety check
                    print(f"Warning: Exceeded expected panel count ({rows * cols})")
                    break
                r, c = divmod(panel_idx, cols)
                ax_i = panel(2 * r,     c)
                ax_c = panel(2 * r + 1, c)

                ax_i.imshow(self._overlay(img_np, hmap.cpu().numpy())) # Assuming _overlay exists
                ax_i.axis("off")
                ax_c.axis("off")
                self._draw_caption(ax_c, text, w, # Assuming _draw_caption exists
                                   max_chars=CAP_WRAP, font_size=CAP_FONT_SIZE)
                panel_idx += 1
            
            if panel_idx >= rows * cols: # Break outer loop if exceeded
                break


        # ----- save / show ------------------------------------------ #
        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
