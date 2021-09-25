#! /usr/bin/env python3
import torch.nn as nn
import time
import torch.optim as optim
import sys
import torch
import matplotlib.pyplot as plt



class CIDA_Train_Eval_Test_Jig:
    def __init__(
        self,
        model:nn.Module,
        label_loss_object,
        domain_loss_object,
        path_to_best_model:str,
        device,
    ) -> None:
        self.model = model.to(device)
        self.label_loss_object = label_loss_object.to(device)
        self.domain_loss_object = domain_loss_object.to(device)
        self.path_to_best_model = path_to_best_model
        self.device = device


    def train(self,
        source_train_iterable,
        source_val_iterable,
        target_train_iterable,
        target_val_iterable,
        num_epochs:int,
        num_logs_per_epoch:int,
        patience:int,
        learning_rate:float,
        alpha_func,
        optimizer_class=optim.Adam
    ):
        last_time = time.time()
        optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)

        # Calc num batches to use and warn if source and target do not match
        num_batches_per_epoch = min(len(source_train_iterable), len(target_train_iterable))
        if len(source_train_iterable) != target_train_iterable:
            print("NOTE: Source and target iterables vary in length ({}, {}). Training only with {} batches per epoch".format(
                    len(source_train_iterable), len(target_train_iterable), num_batches_per_epoch
                )
            )


        logging_decimation_factor = num_batches_per_epoch / num_logs_per_epoch

        for p in self.model.parameters():
            p.requires_grad = True

        history = {}
        history["epoch_indices"] = []
        history["train_label_loss"] = []
        history["train_domain_loss"] = []
        history["source_val_label_loss"] = []
        history["target_val_label_loss"] = []
        history["source_and_target_val_domain_loss"] = []
        history["alpha"] = []

        best_epoch_index_and_val_label_loss = [0, float("inf")]
        for epoch in range(1,num_epochs+1):
            source_train_iter = iter(source_train_iterable)
            target_train_iter = iter(target_train_iterable)

            alpha = alpha_func(epoch, num_epochs)
            
            train_label_loss_epoch = 0
            train_domain_loss_epoch = 0
            num_examples_processed = 0

            for i in range(num_batches_per_epoch):
                self.model.zero_grad()

                """
                Do forward on source
                """
                x,y,t = source_train_iter.next()
                num_examples_processed += x.shape[0]
                x = x.to(self.device)
                y = y.to(self.device)
                t = t.to(self.device)
                y_hat, t_hat = self.model.forward(x, t, alpha)

                # print(t_hat, t)

                source_batch_label_loss = self.label_loss_object(y_hat, y)
                source_batch_domain_loss = self.domain_loss_object(t_hat, t)

                train_label_loss_epoch += source_batch_label_loss.cpu().item()
                train_domain_loss_epoch += source_batch_domain_loss.cpu().item()

                """
                Do forward on target
                """
                x,y,t = target_train_iter.next()
                num_examples_processed += x.shape[0]
                x = x.to(self.device)
                y = y.to(self.device)
                t = t.to(self.device)
                _, t_hat = self.model.forward(x, t, alpha) # Forward on target ignores label because we have no ground truth
                target_batch_domain_loss = self.domain_loss_object(t_hat, t)

                train_domain_loss_epoch += target_batch_domain_loss.cpu().item()

                total_batch_loss = source_batch_label_loss + source_batch_domain_loss + target_batch_domain_loss

                total_batch_loss.backward()
                optimizer.step()

                if i % logging_decimation_factor == 0:
                    cur_time = time.time()
                    examples_per_second =  num_examples_processed / (cur_time - last_time)
                    num_examples_processed = 0
                    last_time = cur_time
                    sys.stdout.write(
                        (
                            "epoch: {epoch}, [iter: {batch} / all {total_batches}], "
                            "examples_per_second: {examples_per_second:.4f}, "
                            "train_label_loss: {train_label_loss:.4f}"
                            "\n"
                        ).format(
                                examples_per_second=examples_per_second,
                                epoch=epoch,
                                batch=i,
                                total_batches=num_batches_per_epoch,
                                train_label_loss=source_batch_label_loss.cpu().item(),
                            )
                    )

                    sys.stdout.flush()

            source_val_acc_label, source_val_label_loss, source_val_domain_loss = self.test(source_val_iterable)
            target_val_acc_label, target_val_label_loss, target_val_domain_loss = self.test(target_val_iterable)

            source_and_target_val_domain_loss = source_val_domain_loss + target_val_domain_loss
            source_and_target_val_label_loss = source_val_label_loss + target_val_label_loss


            history["epoch_indices"].append(epoch)
            history["train_label_loss"].append(train_label_loss_epoch / i)
            history["train_domain_loss"].append(train_domain_loss_epoch / i)
            history["source_val_label_loss"].append(source_val_label_loss)
            history["target_val_label_loss"].append(target_val_label_loss)
            history["source_and_target_val_domain_loss"].append(source_and_target_val_domain_loss)
            history["alpha"].append(alpha)

            sys.stdout.write(
                (
                    "=============================================================\n"
                    "epoch: {epoch}, "
                    "source_val_acc_label: {source_val_acc_label:.4f}, "
                    "target_val_acc_label: {target_val_acc_label:.4f}, "
                    "source_val_label_loss: {source_val_label_loss:.4f}, "
                    "target_val_label_loss: {target_val_label_loss:.4f}, "
                    "source_and_target_val_domain_loss: {source_and_target_val_domain_loss:.4f}"
                    "\n"
                    "=============================================================\n"
                ).format(
                        epoch=epoch,
                        source_val_acc_label=source_val_acc_label,
                        target_val_acc_label=target_val_acc_label,
                        source_val_label_loss=source_val_label_loss,
                        target_val_label_loss=target_val_label_loss,
                        source_and_target_val_domain_loss=source_and_target_val_domain_loss,
                    )
            )

            sys.stdout.flush()

            # New best, save model
            if best_epoch_index_and_val_label_loss[1] > source_and_target_val_label_loss:
                print("New best")
                best_epoch_index_and_val_label_loss[0] = epoch
                best_epoch_index_and_val_label_loss[1] = source_and_target_val_label_loss
                torch.save(self.model, self.path_to_best_model)
            
            # Exhausted patience
            elif epoch - best_epoch_index_and_val_label_loss[0] > patience:
                print("Patience ({}) exhausted".format(patience))
                break
        
        self.history = history

    def test(self, iterable):
        n_batches = 0
        n_total = 0
        n_correct = 0

        total_label_loss = 0
        total_domain_loss = 0

        model = self.model.eval()

        for x,y,t in iter(iterable):
            batch_size = len(x)

            x = x.to(self.device)
            y = y.to(self.device)
            t = t.to(self.device)

            y_hat, t_hat = model(x,t,0) # Forward does not use alpha
            pred = y_hat.data.max(1, keepdim=True)[1]

            n_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            n_total += batch_size

            total_label_loss += self.label_loss_object(y_hat, y).cpu().item()
            total_domain_loss += self.domain_loss_object(t_hat, t).cpu().item()

            n_batches += 1

        accu = n_correct.data.numpy() * 1.0 / n_total
        average_label_loss = total_label_loss / n_batches
        average_domain_loss = total_label_loss / n_batches

        return accu, average_label_loss, average_domain_loss
    
    def show_loss_diagram(self, optional_label_for_loss="Loss"):
        self._do_loss_curve(optional_label_for_loss)
        plt.show()

    def save_loss_diagram(self, path, optional_label_for_loss="Loss"):
        self._do_loss_curve(optional_label_for_loss)
        plt.savefig(path)

    def get_history(self):
        return self.history

    def _do_loss_curve(self, label_for_loss):
        history = self.get_history()

        figure, axis = plt.subplots(1, 1)

        figure.set_size_inches(12, 6)
        figure.suptitle("Loss During Training")
        plt.subplots_adjust(hspace=0.4)
        plt.rcParams['figure.dpi'] = 600
        
        axis.set_title("Label Loss")
        axis.plot(history["epoch_indices"], history['val_label_loss'], label='Validation Label Loss')
        axis.plot(history["epoch_indices"], history['train_label_loss'], label='Train Label Loss')
        axis.legend()
        axis.grid()
        axis.set(xlabel='Epoch', ylabel=label_for_loss)
        axis.locator_params(axis="x", integer=True, tight=True)


if __name__ == "__main__":
    import torch
    import numpy as np
    from cida_images_cnn import CIDA_Images_CNN_Model
    torch.set_default_dtype(torch.float64)

    NUM_CLASSES=16



    NUM_BATCHES = 10000
    SHAPE_DATA = [2,128]
    BATCH_SIZE = 256
    x = np.ones(256*NUM_BATCHES, dtype=np.double)
    x = np.reshape(x, [NUM_BATCHES] + SHAPE_DATA)
    x = torch.from_numpy(x)

    y = np.ones(NUM_BATCHES, dtype=np.double)
    y = torch.from_numpy(y).long()

    dl = torch.utils.data.DataLoader(
        list(zip(x,y)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=50,
        pin_memory=True
    )

    model = CIDA_Images_CNN_Model()
    vanilla_tet_jig = CIDA_Train_Eval_Test_Jig(
        model,
        torch.nn.NLLLoss(),
        "/tmp/model.pb",
        torch.device('cuda')
    )

    vanilla_tet_jig.train(
        train_iterable=dl,
        val_iterable=dl,
        patience=10,
        learning_rate=0.00001,
        num_epochs=20,
        num_logs_per_epoch=5,
    )
    print(vanilla_tet_jig.test(dl))
    print(vanilla_tet_jig.get_history())
    vanilla_tet_jig.show_loss_diagram()

    