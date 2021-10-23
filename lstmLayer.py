class LSTMLayer:
    def __init__(self, n_input, n_cell):
        self.n_input = n_input
        self.n_cell = n_cell
    
    def cellState(self):
        print('cellState')
    
    def forgetGate(self):
        print('forgetGate')
    
    def inputGate(self):
        print('inputGate')
    
    def outputGate(self):
        print('outputGate')