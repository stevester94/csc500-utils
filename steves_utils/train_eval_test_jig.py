#! /usr/bin/env python3
from torch._C import dtype
import torch.nn as nn
import time
import torch.optim as optim
import sys
import torch


class Vanilla_Train_Eval_Test_Jig:
    def __init__(
        self,
        model:nn.Module,
        loss_function,
        path_to_best_dir:str,
        device=torch.device('cpu'),
    ) -> None:
        self.model = model
        self.loss_function = loss_function
        self.path_to_best_dir = path_to_best_dir
        self.device = device


    def train(self,
        train_iterable,
        val_iterable,
        num_epochs:int,
        num_logs_per_epoch:int,
        patience:int,
        learning_rate:float,
        optimizer_class=optim.Adam
    ):
        # training
        best_accu_t = 0.0
        last_time = time.time()
        optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)
        logging_decimation_factor = len(iter(train_iterable)) / num_logs_per_epoch

        history = {}
        history["indices"] = []
        history["source_val_label_loss"] = []
        history["source_val_domain_loss"] = []
        history["target_val_label_loss"] = []
        history["target_val_domain_loss"] = []
        history["source_train_label_loss"] = []
        history["source_train_domain_loss"] = []
        history["source_val_label_accuracy"] = []
        history["target_val_label_accuracy"] = []

        best_epoch_index_and_combined_val_label_loss = [0, float("inf")]
        for epoch in range(1,num_epochs+1):
            train_iter = iter(train_iterable)
            err_label_epoch = 0                

            for i in range(len(train_iter)):

                # training model using source data
                x,y = train_iter.next()

                self.model.zero_grad()

                x = x.to_device(self.device)
                y = y.to_device(self.device)


                y_hat = self.model.forward(x)

                loss = self.loss_function(y_hat, y)

                err_label_epoch += loss.cpu().item()

                err_label_epoch.backward()
                optimizer.step()

                if i % logging_decimation_factor == 0:
                    cur_time = time.time()
                    batches_per_second = logging_decimation_factor / (cur_time - last_time)
                    last_time = cur_time
                    sys.stdout.write(
                        (
                            "epoch: {epoch}, [iter: {batch} / all {total_batches}], "
                            "batches_per_second: {batches_per_second:.4f}, "
                            "err_s_label: {err_s_label:.4f}, "
                            "err_s_domain: {err_s_domain:.4f}, "
                            "alpha: {alpha:.4f}\n"
                        ).format(
                                batches_per_second=batches_per_second,
                                epoch=epoch,
                                batch=i,
                                total_batches=len(source_train_dl),
                                err_s_label=err_s_label.cpu().item(),
                                err_s_domain=err_s_domain.cpu().item(),
                                alpha=alpha
                            )
                    )

                    sys.stdout.flush()

            source_val_label_accuracy, source_val_label_loss, source_val_domain_loss = \
                test(my_net, loss_class, loss_domain, source_val_dl)
            
            target_val_label_accuracy, target_val_label_loss, target_val_domain_loss = \
                test(my_net, loss_class, loss_domain, target_val_dl)

            history["indices"].append(epoch)
            history["source_val_label_loss"].append(source_val_label_loss)
            history["source_val_domain_loss"].append(source_val_domain_loss)
            history["target_val_label_loss"].append(target_val_label_loss)
            history["target_val_domain_loss"].append(target_val_domain_loss)
            history["source_train_label_loss"].append(err_s_label_epoch / i)
            history["source_train_domain_loss"].append(err_s_domain_epoch / i)
            history["source_val_label_accuracy"].append(source_val_label_accuracy)
            history["target_val_label_accuracy"].append(target_val_label_accuracy)

            sys.stdout.write(
                (
                    "=============================================================\n"
                    "epoch: {epoch}, "
                    "acc_src_val_label: {source_val_label_accuracy:.4f}, "
                    "err_src_val_label: {source_val_label_loss:.4f}, "
                    "err_src_val_domain: {source_val_domain_loss:.4f}, "
                    "acc_trgt_val_label: {target_val_label_accuracy:.4f}, "
                    "err_trgt_val_label: {target_val_label_loss:.4f}, "
                    "err_trgt_val_domain: {target_val_domain_loss:.4f}"
                    "\n"
                    "=============================================================\n"
                ).format(
                        epoch=epoch,
                        source_val_label_accuracy=source_val_label_accuracy,
                        source_val_label_loss=source_val_label_loss,
                        source_val_domain_loss=source_val_domain_loss,
                        target_val_label_accuracy=target_val_label_accuracy,
                        target_val_label_loss=target_val_label_loss,
                        target_val_domain_loss=target_val_domain_loss,
                    )
            )

            sys.stdout.flush()

            combined_val_label_loss = source_val_label_loss + target_val_label_loss
            if best_epoch_index_and_combined_val_label_loss[1] > combined_val_label_loss:
                print("New best")
                best_epoch_index_and_combined_val_label_loss[0] = epoch
                best_epoch_index_and_combined_val_label_loss[1] = combined_val_label_loss
                torch.save(my_net, BEST_MODEL_PATH)
            
            elif epoch - best_epoch_index_and_combined_val_label_loss[0] > patience:
                print("Patience ({}) exhausted".format(patience))
                break

    def test(self, iterable):
        n_batches = 0
        n_total = 0
        n_correct = 0
        total_label_loss = 0

        model = self.model.eval()

        for x,y in iter(iterable):
            batch_size = len(x)

            # x = x.to_device(self.device)
            # y = y.to_device(self.device)

            y_hat = model(x=x)

            # print(y_hat.shape)
            # sys.exit(1)

            pred = y_hat.data.max(1, keepdim=True)[1]

            # print(pred)

            n_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            n_total += batch_size
            total_label_loss += self.loss_function(y_hat, y).cpu().item()

            n_batches += 1

        accu = n_correct.data.numpy() * 1.0 / n_total
        average_label_loss = total_label_loss / n_batches

        return accu, average_label_loss
    
    def show_loss_diagram(self):
        pass

    def save_loss_diagram(self):
        pass

    def get_history(self):
        pass


if __name__ == "__main__":
    import torch
    import numpy as np
    torch.set_default_dtype(torch.float64)

    NUM_CLASSES=16

    class CNN_Model(nn.Module):

        def __init__(self):
            super(CNN_Model, self).__init__()

            self.conv = nn.Sequential()
            self.dense = nn.Sequential()

            # Unique naming matters
            
            # This first layer does depthwise convolution; each channel gets (out_channels/groups) number of filters. These are applied, and
            # then simply stacked in the output
            #self.feature.add_module('dyuh_1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1, groups=2))
            self.conv.add_module('dyuh_1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1))
            self.conv.add_module('dyuh_2', nn.ReLU(False)) # Optionally do the operation in place
            self.conv.add_module('dyuh_3', nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2))
            self.conv.add_module('dyuh_4', nn.ReLU(False))
            self.conv.add_module('dyuh_5', nn.Dropout())
            self.conv.add_module("dyuh_6", nn.Flatten())

            self.dense.add_module('dyuh_7', nn.Linear(50 * 58, 80)) # Input shape, output shape
            self.dense.add_module('dyuh_8', nn.ReLU(False))
            self.dense.add_module('dyuh_9', nn.Dropout())
            self.dense.add_module('dyuh_10', nn.Linear(80, NUM_CLASSES))
            self.dense.add_module('dyuh_11', nn.LogSoftmax(dim=1))

        def forward(self, x):
            conv_result = self.conv(x)
            y_hat = self.dense(conv_result)
            return y_hat


    NUM_BATCHES = 1000
    SHAPE_DATA = [2,128]
    BATCH_SIZE = 10
    x = np.arange(256*NUM_BATCHES, dtype=np.double)
    x = np.reshape(x, [NUM_BATCHES] + SHAPE_DATA)
    x = torch.from_numpy(x)

    y = np.ones(NUM_BATCHES, dtype=np.double)
    y = np.reshape(y, [NUM_BATCHES,1])
    y = torch.from_numpy(y)

    # print(x)
    # print(y)
    # print(x_y)

    # print(list(zip(x,y)))

    dl = torch.utils.data.DataLoader(
        list(zip(x,y)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=50,
        pin_memory=True
    )

    model = CNN_Model()
    vanilla_tet_jig = Vanilla_Train_Eval_Test_Jig(
        model,
        torch.nn.L1Loss(),
        "/tmp/"
    )

    vanilla_tet_jig.test(dl)