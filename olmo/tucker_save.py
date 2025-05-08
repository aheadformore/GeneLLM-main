import torch
import tensorly as tl
from tensorly.decomposition import tucker
from transformers import AutoModelForCausalLM





def TuckerExperts(matrices, ranks):
    """
    Tucker decomposition for the expert matrices
    """
    tensor = tl.tensor(matrices.numpy())
    core, factors = tucker(tensor, ranks)
    
    reconstructed_tensor = tl.tucker_to_tensor((core, factors))
    reconstructed_tensor = torch.tensor(reconstructed_tensor)
    
    public_info_matrix = reconstructed_tensor.mean(dim=-1)
    
    return public_info_matrix





# layer2experts = {0 : [0, 21, 40, 41, 17, 53, 54, 6], 1 : [47, 18, 25, 15, 61, 21, 26, 31], 
#                  2 : [60, 61, 45, 40, 5, 14, 30, 28], 3 : [9, 15, 43, 35, 31, 36, 19, 39], 
#                  4 : [21, 17, 25, 6, 19, 55, 4, 61], 5 : [42, 0, 2, 53, 31, 17, 8, 21], 
#                  6 : [52, 57, 18, 42, 24, 5, 1, 62], 7 : [17, 4, 2, 35, 58, 45, 21, 0], 
#                  8 : [54, 16, 27, 3, 37, 14, 45, 51], 9 : [51, 46, 5, 4, 7, 6, 20, 52], 
#                  10 : [56, 43, 44, 11, 2, 62, 13, 19], 11 : [47, 27, 13, 23, 44, 7, 46, 33], 
#                  12 : [38, 43, 51, 31, 55, 8, 45, 59], 13 : [2, 25, 62, 32, 20, 61, 7, 6],
#                  14 : [58, 6, 36, 7, 39, 9, 11, 24], 15 : [17, 30, 58, 36, 8, 1, 44, 29]}





layer2experts = {0 : [0, 21, 40, 41], 1 : [47, 18, 25, 15], 
                 2 : [60, 61, 45, 40], 3 : [9, 15, 43, 35], 
                 4 : [21, 17, 25, 6], 5 : [42, 0, 2, 53], 
                 6 : [52, 57, 18, 42], 7 : [17, 4, 2, 35], 
                 8 : [54, 16, 27, 3], 9 : [51, 46, 5, 4], 
                 10 : [56, 43, 44, 11], 11 : [47, 27, 13, 23], 
                 12 : [38, 43, 51, 31], 13 : [2, 25, 62, 32],
                 14 : [58, 6, 36, 7], 15 : [17, 30, 58, 36]}






# layer2experts = {0 : [0, 21], 1 : [47, 18], 
#                  2 : [60, 61], 3 : [9, 15], 
#                  4 : [21, 17], 5 : [42, 0], 
#                  6 : [52, 57], 7 : [17, 4], 
#                  8 : [54, 16], 9 : [51, 46], 
#                  10 : [56, 43], 11 : [47, 27], 
#                  12 : [38, 43], 13 : [2, 25],
#                  14 : [58, 6], 15 : [17, 30]}



















if __name__ == '__main__':
    # Load the model
    model = AutoModelForCausalLM.from_pretrained('/home/liuchang/model/OLMoE')
    ancestor_state_dict = model.state_dict()
    num_experts_to_tuckered = 4
    
    # remain the experts activated the most, delete the others
    # del ancestor_state_dict['lm_head.weight']
    for key in list(ancestor_state_dict.keys()):
        if "experts" in key:
            layer_id = int(key.split(".")[2])
            expert_id = int(key.split(".")[5])
            if expert_id not in layer2experts[layer_id]:
                del ancestor_state_dict[key]
        # if "gate.weight" in key:
        else:
            del ancestor_state_dict[key]
    
    state_to_load = ancestor_state_dict
    
    

    # re-ID the experts
    for i in range(16):
        layer2experts[i].sort()
        for j in range(num_experts_to_tuckered):
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
            
    print(state_to_load.keys())


   # Tucker decomposition
    print("Tucker decomposition starts.")
    for l in range(16):
        print("Layer {} tucker decomposition...".format(l))
        m_gate = []
        m_up = []
        m_down = []
        for k in range(num_experts_to_tuckered):
            m_gate.append(state_to_load['model.layers.{}.mlp.experts.{}.gate_proj.weight'.format(l, k)])
            m_up.append(state_to_load['model.layers.{}.mlp.experts.{}.up_proj.weight'.format(l, k)])
            m_down.append(state_to_load['model.layers.{}.mlp.experts.{}.down_proj.weight'.format(l, k)])
            
            del state_to_load['model.layers.{}.mlp.experts.{}.gate_proj.weight'.format(l, k)]
            del state_to_load['model.layers.{}.mlp.experts.{}.up_proj.weight'.format(l, k)]
            del state_to_load['model.layers.{}.mlp.experts.{}.down_proj.weight'.format(l, k)]
            
        m_gate = torch.stack(m_gate, dim=-1)
        m_up = torch.stack(m_up, dim=-1)
        m_down = torch.stack(m_down, dim=-1)
        
        # print(m_gate.shape, m_up.shape, m_down.shape)
        
        
        
        # # tucker decomposition
        # gate_rank = [512, 1024, 2]
        # up_rank = [512, 1024, 2]
        # down_rank = [512, 1024, 2]
        
        # gate_tucker = TuckerExperts(m_gate, gate_rank)
        # up_tucker = TuckerExperts(m_up, up_rank)
        # down_tucker = TuckerExperts(m_down, down_rank)
        
        # state_to_load['model.layers.{}.mlp.experts.0.gate_proj.weight'.format(l)] = gate_tucker
        # state_to_load['model.layers.{}.mlp.experts.0.up_proj.weight'.format(l)] = up_tucker
        # state_to_load['model.layers.{}.mlp.experts.0.down_proj.weight'.format(l)] = down_tucker
        
        # average 
        gate_average = m_gate.mean(dim=-1)
        up_average = m_up.mean(dim=-1)
        down_average = m_down.mean(dim=-1)
        
        
        state_to_load['model.layers.{}.mlp.experts.0.gate_proj.weight'.format(l)] = gate_average
        state_to_load['model.layers.{}.mlp.experts.0.up_proj.weight'.format(l)] = up_average
        state_to_load['model.layers.{}.mlp.experts.0.down_proj.weight'.format(l)] = down_average
            
            
    print("Tucker decomposition finished.")    
    
    # Save the model
    torch.save(state_to_load, '/home/liuchang/model/olmoe_tucker/4_1_average.pt')     
    print("Model saved.")















