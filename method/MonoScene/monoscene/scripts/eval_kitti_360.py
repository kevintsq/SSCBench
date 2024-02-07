from pytorch_lightning import Trainer
from monoscene.models.monoscene import MonoScene
from monoscene.data.kitti_360.kitti_360_dm import Kitti360DataModule
import torch


def main():
    torch.set_grad_enabled(False)
    feature = 64
    project_scale = 2
    full_scene_size = (256, 256, 32)
    data_module = Kitti360DataModule(
        root=r"/mnt/hdd_0/SSCBenchAssets/datasets/kitti360",
        preprocess_root=r"/mnt/hdd_0/SSCBenchAssets/datasets/kitti360/preprocess/unified",
        frustum_size=8,
        batch_size=1
    )

    trainer = Trainer(
        sync_batchnorm=True, accelerator="gpu"
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
    val_dataloader = data_module.val_dataloader()
    trainer.test(model, dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
