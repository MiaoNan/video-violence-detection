"""Training entry point. The public main(config) function remains available."""
from training.runner import main


if __name__ == '__main__':
    import torch
    from config import Config

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = Config(
        'densenet_lean',  # only factory included in this checkout
        'cctv',
        device=device,
        num_epoch=150,
        acc_baseline=0.92,
        ft_begin_idx=0,
    )
    config.num_cv = 0

    # train params for different datasets
    configs = {
        'hockey': {
            'lr': 1e-2,
            'batch_size': 32
        },
        'movie': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'vif': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'cctv': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'mix': {
            'lr': 1e-3,
            'batch_size': 10
        }
    }

    for dataset in ['cctv']:
        config.dataset = dataset
        config.train_batch = configs[dataset]['batch_size']
        config.val_batch = configs[dataset]['batch_size']
        config.learning_rate = configs[dataset]['lr']
        main(config)
        # # 5 fold cross validation
        # for cv in range(1, 6):
        #     config.num_cv = cv
        #     main(config)
