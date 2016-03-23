import os
from AlphaGo.training.supervised_policy_trainer import supervised_policy_trainer
from AlphaGo.models.policy import CNNPolicy
import unittest

class TestSupervisedPolicyTrainer(unittest.TestCase):
    def testTrain(self):
        net = CNNPolicy.create_network(input_dim=46)
        trainer = supervised_policy_trainer(train_batch_size=3,nb_epoch=2)
        trainer.train(net,train_folder='tests/test_data/pickle/train',
                          test_folder = 'tests/test_data/pickle/test',
                          model_folder = 'tests/test_data',
                          checkpt_prefix="test")
        #TODO: What's a good check that training has been conducted as expected?

        # Remove files created during testing
        [os.remove(os.path.join("tests/test_data",f)) for
            f in os.listdir("tests/test_data")
                if f.split('.')[0]=="test"]

if __name__=='__main__':
    unittest.main()
