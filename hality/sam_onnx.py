from __future__ import annotations

import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "sam", "sam_vit_b.pth")
ONNX = os.path.join(ROOT, "models", "sam", "sam_encoder_vit_b.onnx")


def exportar(ckpt: str = CKPT, saida: str = ONNX, tipo: str = "vit_b") -> str:
    from segment_anything import sam_model_registry
    sam = sam_model_registry[tipo](checkpoint=ckpt)
    sam.eval()

    class Encoder(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, x):
            return self.enc(x)

    os.makedirs(os.path.dirname(saida), exist_ok=True)
    dummy = torch.randn(1, 3, sam.image_encoder.img_size, sam.image_encoder.img_size)
    with torch.no_grad():
        torch.onnx.export(
            Encoder(sam.image_encoder), dummy, saida,
            input_names=["imagem"], output_names=["features"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    return saida


class SamGPU:

    def __init__(self, ckpt: str = CKPT, onnx: str = ONNX, tipo: str = "vit_b") -> None:
        import onnxruntime as ort
        from segment_anything import SamPredictor, sam_model_registry

        if not os.path.exists(onnx):
            raise FileNotFoundError(
                f"{onnx} nao existe. Rode: python -m hality.sam_onnx --exportar")

        provedores = ort.get_available_providers()
        self.provedor = "DmlExecutionProvider" if "DmlExecutionProvider" in provedores \
            else "CPUExecutionProvider"
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sessao = ort.InferenceSession(onnx, opts, providers=[self.provedor])

        sam = sam_model_registry[tipo](checkpoint=ckpt)
        sam.eval()
        torch.set_num_threads(os.cpu_count() or 4)
        self.pred = SamPredictor(sam)
        self.lado = sam.image_encoder.img_size

    def _preparar(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        from segment_anything.utils.transforms import ResizeLongestSide
        tr = ResizeLongestSide(self.lado)
        x = tr.apply_image(rgb)
        entrada = x.shape[:2]
        t = torch.as_tensor(x).permute(2, 0, 1).contiguous()[None].float()
        t = (t - self.pred.model.pixel_mean) / self.pred.model.pixel_std
        h, w = t.shape[-2:]
        t = torch.nn.functional.pad(t, (0, self.lado - w, 0, self.lado - h))
        return t.numpy(), entrada

    def set_image(self, rgb: np.ndarray) -> None:
        entrada_np, entrada = self._preparar(rgb)
        feats = self.sessao.run(["features"], {"imagem": entrada_np})[0]
        self.pred.features = torch.from_numpy(feats)
        self.pred.original_size = rgb.shape[:2]
        self.pred.input_size = entrada
        self.pred.is_image_set = True

    def isolar(self, rgb: np.ndarray, semente: np.ndarray | None = None,
               seed: int = 0) -> np.ndarray | None:
        from .sam_segmenter import LADO, gerar_prompts
        from PIL import Image

        rng = np.random.default_rng(seed)
        H0, W0 = rgb.shape[:2]
        esc = LADO / max(H0, W0)
        if esc < 1:
            peq = np.asarray(Image.fromarray(rgb).resize(
                (int(W0 * esc), int(H0 * esc)), Image.BICUBIC))
            sem = None if semente is None else np.asarray(Image.fromarray(
                (semente.astype(np.uint8) * 255)).resize(
                peq.shape[1::-1], Image.NEAREST)) > 127
        else:
            peq, sem = rgb, semente

        p = gerar_prompts(peq, sem, rng)
        if p is None:
            return None
        pontos, rotulos = p

        self.set_image(peq)
        masks, scores, _ = self.pred.predict(point_coords=pontos, point_labels=rotulos,
                                             multimask_output=True)
        melhor, nota_melhor = None, -1e9
        for m, s in zip(masks, scores):
            a = m.mean()
            if not (0.05 < a < 0.85):
                continue
            nota = float(s) - 2.0 * abs(a - 0.40)
            if nota > nota_melhor:
                melhor, nota_melhor = m, nota
        if melhor is None:
            melhor = masks[int(np.argmax(scores))]
        if melhor.shape != (H0, W0):
            melhor = np.asarray(Image.fromarray((melhor.astype(np.uint8) * 255)).resize(
                (W0, H0), Image.NEAREST)) > 127
        return melhor


if __name__ == "__main__":
    import sys
    if "--exportar" in sys.argv:
        print("exportando o encoder para ONNX (alguns minutos)...", flush=True)
        p = exportar()
        print("salvo em", p, "-", os.path.getsize(p) // 1024 // 1024, "MB")
    else:
        import time
        s = SamGPU()
        print("provedor em uso:", s.provedor)
        rgb = np.random.default_rng(0).integers(0, 256, (720, 960, 3), dtype=np.uint8)
        s.set_image(rgb)
        t = time.time()
        for _ in range(3):
            s.set_image(rgb)
        print("encoder: %.2f s/imagem" % ((time.time() - t) / 3))
