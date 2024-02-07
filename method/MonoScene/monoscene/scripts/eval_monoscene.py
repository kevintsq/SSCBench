from pytorch_lightning import Trainer
from monoscene.models.monoscene import MonoScene
from monoscene.data.nuscenes.nuscenes_dm import NuScenesDataModule
import torch


def main():
    torch.set_grad_enabled(False)
    full_scene_size = (256, 256, 32)
    project_scale = 2
    feature = 64
    data_module = NuScenesDataModule(
        root=r"/mnt/hdd_0/SSCBenchAssets/datasets/nuscenes_trans",
        preprocess_root=r"/mnt/hdd_0/SSCBenchAssets/datasets/nuscenes_trans/preprocess_uni",
        frustum_size=8,
        batch_size=1,
    )

    trainer = Trainer(
        sync_batchnorm=True, deterministic="warn", accelerator="gpu"
    )

    model_path = r"/mnt/hdd_0/SSCBenchAssets/ckpts/monoscene_nuscenes_uni.ckpt"

    model = MonoScene.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=True,
        full_scene_size=full_scene_size,
    )
    model.eval()
    data_module.setup()
    test_dataloader = data_module.test_dataloader()
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
