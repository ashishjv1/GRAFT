from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str
    dataset_name: str
    num_epochs: int 
    batch_size: int
    lr: float
    device: str
    optimizer: str
    weight_decay: float
    grad_clip: float
    fraction: float
    selection_iter: int
    warm_start: bool
    sched: str = "cosine"
    num_workers: int = 4
    
    @classmethod
    def from_args(cls, args):
        return cls(
            model_name=args.model,
            dataset_name=args.dataset,
            num_epochs=args.numEpochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            fraction=args.fraction,
            selection_iter=args.select_iter,
            warm_start=args.warm_start,
            sched=getattr(args, 'sched', 'cosine')
        )
