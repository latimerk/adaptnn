from adaptnn.retina_datasets import NiruDataset
from adaptnn.visual_neuron_convnets import PopulationConvNet, penalize_convnet_weights
import torch

class NiruDataModel:
    def __init__(self, dataset_params = {"recording" : "A-noise"},
                       net_params = {"out_activation" : torch.nn.Softplus}):

        self.dataset = NiruDataset(**dataset_params)

        self.model = PopulationConvNet(num_cells=self.dataset.num_cells,
                                       frame_width=self.dataset.frame_width,
                                       frame_width=self.dataset.frame_height,
                                       **net_params)
        
        self.dataset.set_timepadding(self.model.time_padding)


    def predict(self):
        with torch.no_grad():
            X_test, Y_test = self.dataset.get_test()
            return self.model(X_test), Y_test
        

    def get_loss_function(self, **kwargs):
        return torch.nn.PoissonNLLLoss(log_input=self.model.nonlinear_output, **kwargs)
    
    def get_penalty_function(self,  en_lambda = 0.01,
                                    en_alpha = 0.5,
                                    fl_lambda_x = 0.01,
                                    fl_lambda_y = 0.01,
                                    fl_lambda_t = 0.01,
                                    lin_en_lambda = 0.01,
                                    lin_en_alpha = 0.5):
        return lambda mm : penalize_convnet_weights(mm, en_lambda, en_alpha,
                                                        fl_lambda_x, fl_lambda_y, fl_lambda_t,
                                                        lin_en_lambda, lin_en_alpha)
    
    def train(self,
              epochs : int,
              optimizer_params = {"lr" : 1e-4},
              scheduler_params = {"start_factor" : 1.0, "end_factor" : 0.1, "total_iters" : 2000},
              batch_params = {"batch_size":16, "shuffle":True},
              penalty_params = {},
              loss_params = {}):

        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)

        penalty = self.get_penalty_function(**penalty_params)
        criterion = self.get_loss_function(**loss_params)

        train_dataloader = torch.utils.data.DataLoader(self.dataset, batch_params)

        for epoch in range(epochs):
            running_loss = 0
            for X_t, Y_t in enumerate(train_dataloader):
        
                #convert numpy array to torch Variable
                optimizer.zero_grad()
                
                #Forward to get outputs
                outputs=self.model(X_t)
                
                #calculate loss
                loss=criterion(outputs, Y_t) + penalty(self.model)
                
                #getting gradients wrt parameters
                loss.backward()
                
                #updating parameters
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                
            if((epoch+1) % 50 == 0):
                print(f"epoch {epoch+1}, loss {running_loss}, step size {optimizer.param_groups[0]['lr']}")