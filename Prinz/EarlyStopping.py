class EarlyStopping:
    def __init__(self,p=0):
        self.patience=p
        self.j=0
        self.v=inf
        self.other_parameters=None

    def reset(self):
        self.j=0
        self.v=inf
        self.other_parameters=None
    
    def read_validation_result(self,model,validation_cost,other_parameters=None):
        if validation_cost<self.v:
            self.j=0
            self.model=copy.deepcopy(model)
            self.v=validation_cost
            self.other_parameters=other_parameters
        else:
            self.j+=1
        if self.j>=self.patience:
            return True
        return False
    
    def get_best_model(self):
        return copy.deepcopy(self.model)

    def get_best_other_parameters(self):
        return self.other_parameters
