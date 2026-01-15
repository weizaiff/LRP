






def get_mask_neuron_model_vllm_LRP(model, activation_mask_dict, is_llama = True):


        
        if activation_mask_dict:
            def factory(mask):
                def llama_forward_lrp(self, x):
                    '''
                        mask: {'up_proj':[...], 'gate_proj':[...], 'down_proj':[...]}
                        
                    '''
                    gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                    i = gate_up.size(-1)
                    activation = F.silu(gate_up[:, :, : i // 2])
                    if 'gate_proj' in mask:
                        activation.index_fill_(2, mask['gate_proj'], 0)

                    if 'up_proj' in mask:
                        x = activation * gate_up[:, :, i // 2 :].index_fill_(2, mask['up_proj'], 0)
                    else:
                        x = activation * gate_up[:, :, i // 2 :]
                    x, _ = self.down_proj(x)
                    if 'down_proj' in mask:
                        x.index_fill_(2, mask['down_proj'], 0)
                    return x
                def llama_forward(self, x):
                    gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                    i = gate_up.size(-1)
                    activation = F.silu(gate_up[:, :, : i // 2])
                    activation.index_fill_(2, mask, 0)
                    x = activation * gate_up[:, :, i // 2 :]
                    x, _ = self.down_proj(x)
                    return x
                def llama_forward_split(self, x):
                    gate_ = self.gate_proj(x)  # b, l, 2i
                    i = gate_.size(-1)
                    activation = F.silu(gate_)
                    #activation.index_fill_(2, mask, 0)

                    # test
                    print(activation.shape)
                    activation.index_fill_(2, torch.tensor(list(range(activation.shape[1]))), 0 )
                    x = activation * self.up_proj(x)
                    x = self.down_proj(x)
                    return x
    
                def bloom_forward(self, x: torch.Tensor):
                    x, _ = self.dense_h_to_4h(x)
                    x = self.gelu_impl(x)
                    x.index_fill_(2, mask, 0)
                    x, _ = self.dense_4h_to_h(x)
                    return x
    
                if is_llama:
                    return llama_forward_lrp
                else:
                    return bloom_forward
    
            for i, ilayer_mask_dict in enumerate(activation_mask_dict): 
                #print('ilayer:',i, layer_mask)
                #if is_llama:
                    # latest
                    #obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
                #else:
                    # latest
                    #obj = model.llm_engine.model_executor.driver_worker.model_runner.model.transformer.h[i].mlp

                if is_llama:
                    obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
                else:
                    obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp

                ilayer_mask_dict_cuda = {}
                for ikey ,ival in ilayer_mask_dict.items():
                    if len(ival)>0:
                        ilayer_mask_dict_cuda[ikey] = torch.LongTensor(ival).to('cuda')
                
                
                obj.forward = MethodType(factory(ilayer_mask_dict_cuda), obj)
        return model