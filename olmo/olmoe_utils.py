import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import OlmoeForCausalLM, OlmoeModel, OlmoePreTrainedModel, AutoModelForCausalLM, OlmoeConfig LlamaForSequenceClassification
import torch.nn as nn
from enum import Enum
import tensorly as tl
from tensorly.decomposition import tucker
from olmo.config import TrainConfig
# from .model_olmoe import OlmoeForSequenceClassification

class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4


AUTO_MODELS = {
    # TaskType.TOKEN_CLASSIFICATION: BertForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: OlmoeForSequenceClassification,
    # TaskType.QUESTION_ANSWERING: BertForQuestionAnswering,
    # TaskType.MULTIPLE_CHOICE: BertForMultipleChoice,
}










layer2experts = {0 : [0, 21, 40, 41, 17, 53, 54, 6], 1 : [47, 18, 25, 15, 61, 21, 26, 31], 
                 2 : [60, 61, 45, 40, 5, 14, 30, 28], 3 : [9, 15, 43, 35, 31, 36, 19, 39], 
                 4 : [21, 17, 25, 6, 19, 55, 4, 61], 5 : [42, 0, 2, 53, 31, 17, 8, 21], 
                 6 : [52, 57, 18, 42, 24, 5, 1, 62], 7 : [17, 4, 2, 35, 58, 45, 21, 0], 
                 8 : [54, 16, 27, 3, 37, 14, 45, 51], 9 : [51, 46, 5, 4, 7, 6, 20, 52], 
                 10 : [56, 43, 44, 11, 2, 62, 13, 19], 11 : [47, 27, 13, 23, 44, 7, 46, 33], 
                 12 : [38, 43, 51, 31, 55, 8, 45, 59], 13 : [2, 25, 62, 32, 20, 61, 7, 6],
                 14 : [58, 6, 36, 7, 39, 9, 11, 24], 15 : [17, 30, 58, 36, 8, 1, 44, 29]}




# layer2experts = {0 : [0, 21], 1 : [47, 18], 
#                  2 : [60, 61], 3 : [9, 15], 
#                  4 : [21, 17], 5 : [42, 0], 
#                  6 : [52, 57], 7 : [17, 4], 
#                  8 : [54, 16], 9 : [51, 46], 
#                  10 : [56, 43], 11 : [47, 27], 
#                  12 : [38, 43], 13 : [2, 25],
#                  14 : [58, 6], 15 : [17, 30]}



# layer2experts = {0 : [0], 1 : [47], 
#                  2 : [60], 3 : [9], 
#                  4 : [21], 5 : [42], 
#                  6 : [52], 7 : [17], 
#                  8 : [54], 9 : [51], 
#                  10 : [56], 11 : [47], 
#                  12 : [38], 13 : [2],
#                  14 : [58], 15 : [17]}













def get_olmoe_model(train_config: TrainConfig):
    
    if train_config.tuckered_ancestor_path:
        state_to_load = torch.load(train_config.tuckered_ancestor_path)
        olmoeconfig = OlmoeConfig(intermediate_size=1024, num_experts=1, num_experts_per_tok=1)    
        model = OlmoeForCausalLM(olmoeconfig)
        model.load_state_dict(state_to_load, strict=False)
        print("Tuckered experts have been loaded to the model.")
        return model
    
    
    
    
    else:
        # load model parameters and only retain the experts chosen
        ancestor_model = AutoModelForCausalLM.from_pretrained(
            train_config.ancestor_path)
        ori_model_state_from_huggingface = ancestor_model.state_dict()
        
        

        # remain the experts activated the most, delete the others
        del ori_model_state_from_huggingface['lm_head.weight']
        for key in list(ori_model_state_from_huggingface.keys()):
            if "experts" in key:
                layer_id = int(key.split(".")[2])
                expert_id = int(key.split(".")[5])
                if expert_id not in layer2experts[layer_id]:
                    del ori_model_state_from_huggingface[key]
            # if "gate.weight" in key:
            else:
                del ori_model_state_from_huggingface[key]
        
        state_to_load = ori_model_state_from_huggingface
        

        

        # re-ID the experts
        for i in range(16):
            layer2experts[i].sort()
            for j in range(8):
                old_gate_proj = 'model.layers.{}.mlp.experts.{}.gate_proj.weight'.format(i, layer2experts[i][j])
                old_up_proj = 'model.layers.{}.mlp.experts.{}.up_proj.weight'.format(i, layer2experts[i][j])
                old_down_proj = 'model.layers.{}.mlp.experts.{}.down_proj.weight'.format(i, layer2experts[i][j])
                
                gate_proj = state_to_load[old_gate_proj]
                del state_to_load[old_gate_proj]
                state_to_load['model.layers.{}.mlp.experts.{}.gate_proj.weight'.format(i, j)] = gate_proj
                
                up_proj = state_to_load[old_up_proj]
                del state_to_load[old_up_proj]
                state_to_load['model.layers.{}.mlp.experts.{}.up_proj.weight'.format(i, j)] = up_proj
                
                down_proj = state_to_load[old_down_proj]
                del state_to_load[old_down_proj]
                state_to_load['model.layers.{}.mlp.experts.{}.down_proj.weight'.format(i, j)] = down_proj
                
                
            
        olmoeconfig = OlmoeConfig(intermediate_size=1024, num_experts=1, num_experts_per_tok=1, num_hidden_layers=1)    
        model = OlmoeForCausalLM(olmoeconfig)
        
        # load the model parameters
        model.load_state_dict(state_to_load, strict=False)
        
        
        param_to_update = 0
        
        # if fix_expert:
        #     for name, param in model.named_parameters():
        #         if 'gate.weight' in name or 'classifier.weight' in name or 'classifier.bias' in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
                    
        # s = model.state_dict()

        param_to_update = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('***** The number of param to be updated is {} *****'.format(param_to_update))
        
        return model
        

































