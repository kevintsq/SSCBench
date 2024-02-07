from monoscene.data.NYU.nyu_dm import NYUDataModule
from monoscene.data.semantic_kitti.kitti_dm import KittiDataModule
from monoscene.data.kitti_360.kitti_360_dm_vis import Kitti360DataModule
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_name="../config/monoscene.yaml")
def main(config: DictConfig):
    # Setup dataloader
    if config.dataset == "kitti" or config.dataset == "kitti_360":
        if config.dataset == "kitti":
            data_module = KittiDataModule(
                root=config.kitti_root,
                preprocess_root=config.kitti_preprocess_root,
                frustum_size=config.frustum_size,
                batch_size=int(config.batch_size / config.n_gpus),
                num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            )
            data_module.setup()
            data_loader = data_module.val_dataloader()
            # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
        else:
            data_module = Kitti360DataModule(
                root=config.kitti_360_root,
                sequences=[config.kitti_360_sequence],
                n_scans=2000,
                batch_size=1,
                num_workers=0,
            )
            data_module.setup()
            data_loader = data_module.dataloader()

    elif config.dataset == "NYU":
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )
        data_module.setup()
        data_loader = data_module.val_dataloader()
        # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
    else:
        print("dataset not support")

    for batch in tqdm(data_loader):
        pass

if __name__ == "__main__":
    main()
