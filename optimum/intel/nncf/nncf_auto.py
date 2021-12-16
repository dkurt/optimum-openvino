import os

from nncf import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.torch.initialization import PTInitializingDataLoader


def get_train_dataloader_for_init(args, train_dataset, data_collator=None):
    from torch.utils.data import RandomSampler
    from torch.utils.data import DistributedSampler
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    if data_collator is None:
        from transformers.data.data_collator import default_data_collator
        data_collator = default_data_collator

    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=args.dataloader_drop_last,
    )
    return data_loader


class NNCFAutoConfig(NNCFConfig):
    def auto_register_extra_structs(self, args, train_dataset, data_collator):
        if self.get("log_dir") is None:
            self["log_dir"] = args.output_dir
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(self["log_dir"])
        if args.do_train:
            train_dataloader = get_train_dataloader_for_init(args, train_dataset, data_collator)
            class SquadInitializingDataloader(PTInitializingDataLoader):
                def get_inputs(self, dataloader_output):
                    return (), dataloader_output

            self.register_extra_structs([
                QuantizationRangeInitArgs(SquadInitializingDataloader(train_dataloader)),
                BNAdaptationInitArgs(SquadInitializingDataloader(train_dataloader)),
            ])
