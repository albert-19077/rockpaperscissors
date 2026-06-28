import os
import torch
import config
from model import RockPaperScissorsCNN


def export_model():
    weights_path = config.MODEL_PATH
    if not os.path.exists(weights_path) and os.path.exists(f"{weights_path}.pth"):
        weights_path = f"{weights_path}.pth"
    if not os.path.exists(weights_path):
        print(f"Hata: Model ağırlık dosyası '{weights_path}' bulunamadı. Lütfen önce modeli eğitin.")
        return
    model = RockPaperScissorsCNN(num_classes=config.NUM_CLASSES)
    try:
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        print("Model ağırlıkları yüklendi.")
    except Exception as e:
        print(f"Hata: Ağırlıklar yüklenemedi ({e}).")
        return
    model.eval()

    # Giriş Şekli: (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
    onnx_path = config.ONNX_MODEL_NAME
    print(f"ONNX formatına aktarılıyor: {onnx_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"ONNX dönüşümü başarılı! Dosya: {os.path.abspath(onnx_path)}")
    except Exception as e:
        print(f"ONNX aktarım hatası: {e}")


if __name__ == "__main__":
    export_model()
