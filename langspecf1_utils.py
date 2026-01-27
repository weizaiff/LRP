import json


def get_analysis_res(ds_judge_res):
    df_judge_res = ds_judge_res['train'].to_pandas()
    # get score
    def get_score(x):
        try:
            return  json.loads(x)['score']
        except:
            return -1
    
    df_judge_res['score'] = df_judge_res['output'].apply(get_score)

    df_judge_res['class_tpye'] = df_judge_res['lang']+'|' + df_judge_res['model_type']

    result_dict = {}
    for imodel_type in list(df_judge_res['class_tpye'].value_counts().keys()):
        print('='*20)
        print('imodel_type:', imodel_type)
        df_tmp = df_judge_res.loc[df_judge_res['class_tpye'] == imodel_type].reset_index(drop=True)
        df_tmp = df_tmp.loc[df_tmp['score']!=-1].reset_index(drop=True)
        print('valid count:', len(df_tmp))
        result_dict[imodel_type] = df_tmp['score'].mean()
              
    return result_dict, df_judge_res



def get_mask_neuron_lang(class_type):
    tmp_list = class_type.split('|')
    if tmp_list[1]=='org_model':
        return 'no_mask'
    
    mid_name_list = tmp_list[1].split('_')
    count = int('en' in mid_name_list) + int('vi' in mid_name_list) + int('zh' in mid_name_list) 
    assert count == 1,(class_type, count)
    if 'en' in mid_name_list:
        mid_name_list.remove('en')
        return 'en'
    if 'vi' in mid_name_list:
        mid_name_list.remove('vi')
        return 'vi'
    if 'zh' in mid_name_list:
        mid_name_list.remove('zh')
        return 'zh'

        

def get_method_name(class_type):
    tmp_list = class_type.split('|')
    if tmp_list[1]=='org_model':
        return 'no_mask'
        
    mid_name_list = tmp_list[1].split('_')
    count = int('en' in mid_name_list) + int('vi' in mid_name_list) + int('zh' in mid_name_list) 
    
    if 'en' in mid_name_list:
        mid_name_list.remove('en')
    if 'vi' in mid_name_list:
        mid_name_list.remove('vi')
    if 'zh' in mid_name_list:
        mid_name_list.remove('zh')

    return '_'.join(mid_name_list)



def calc_metric(df_judge_res,result_dict, beta = 1 ):
    '''
        calc LPS:
            LPS = drop(target) / mean(drop(other)) * mean(drop)
    
        20251215 new metric:
            sigmoid(alpha*exp(drop_target) - (1-alpha)*exp(drop_other))
    
        20251217:
            增加一个版本:
                使用sigmoid(alpha*exp(drop_target_rate) - (1-alpha)*exp(drop_other_rate))计算NLU的一些指标
    
        20251218:
         precision = target_drop / (target_drop + max_other_drop + EPS)
                recall = target_drop / (org_model_task_result[target_task] + EPS)
                 f1 = 2 * precision * recall / (precision + recall + EPS)
    
                *_drop = max(org_drop, 0)#只考虑指标掉了的情况，如果指标增加了，说明没有找到暂时不考虑
        20260121:
            
                precision = target_drop / (target_drop + sum(all_other_drop) + EPS)
                recall = target_drop / (org_model_task_result[target_task] + EPS)
                 f1 = 2 * precision * recall / (precision + recall + EPS)
    
                *_drop = max(org_drop, 0)#只考虑指标掉了的情况，如果指标增加了，说明没有找到暂时不考虑    
    '''
    
    # get org
    df_judge_res['specific_method'] = df_judge_res['class_tpye'].apply(lambda x: x.split('|')[1])
    df_org = df_judge_res.loc[df_judge_res['specific_method'] =='org_model' ]
    
    # 获取不同语种测试集合的分数
    def get_all_lang_score(df_org):
        lang_list = ['en', 'vi', 'zh']
        lang_score_baseline = {}
        for ilang in lang_list:
            lang_score_baseline[ilang] = df_org.loc[df_org['lang']==ilang]['score'].mean()
        return lang_score_baseline
    # baseline
    lang_score_baseline = get_all_lang_score(df_org)
    
    # calc LPS:
    res_score = {}
    res_acuall_score = {}
    for iexp_name in result_dict.keys():
        specific_method = iexp_name.split('|')[1]
        if specific_method=='org_model':continue
    
        df_tmp = df_judge_res.loc[df_judge_res['specific_method']==specific_method ].reset_index(drop=True)
        lang_score_tmp = get_all_lang_score(df_tmp)
    
        assert len(dict(df_tmp['mask_neuron_lang'].value_counts()))==1 and len(dict(df_tmp['method_name'].value_counts()))==1
    
       
    
        assert df_tmp['mask_neuron_lang'][0] not in res_score,(specific_method, df_tmp['mask_neuron_lang'][0], res_score)
    
        if df_tmp['method_name'][0] not in res_score:
            res_score[df_tmp['method_name'][0]] = {}
            res_acuall_score[df_tmp['method_name'][0]] = {}
             
        # calc per lang score
        tmp_drop = {}
        for ikey in lang_score_baseline.keys():
            # drop
            #tmp_drop[ikey] = lang_score_baseline[ikey] - lang_score_tmp[ikey]
            # drop rate
            #tmp_drop[ikey] = (lang_score_baseline[ikey] - lang_score_tmp[ikey])/lang_score_baseline[ikey]
            
            tmp_drop[ikey] = max(lang_score_baseline[ikey] - lang_score_tmp[ikey], 0)
    
        tmp_taget_lang = df_tmp['mask_neuron_lang'][0]
        lang_list = ['en', 'vi', 'zh']
        lang_list.remove(tmp_taget_lang)
    
        # LPS
        #LPS = drop(target) / mean(drop(other)) * mean(drop)
        #LPS = tmp_drop[tmp_taget_lang]/((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2)*(sum(tmp_drop.values())/3)
        #res_score[df_tmp['method_name'][0]][tmp_taget_lang] = LPS
    
        # sigmoid LPS
        k = 2
        f = lambda x: 2/(1+np.exp(-k*x))
        # sigmoid(drop_target) /sigmoid(mean_drop_other) * sigmoid(all_mean_drop) 
        
        #LPS_sig = f(tmp_drop[tmp_taget_lang]) / f(((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2)) #* f(sum(tmp_drop.values())/3)
        
        #res_score[df_tmp['method_name'][0]][tmp_taget_lang] = LPS_sig
    
        #res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] = lang_score_tmp
    
    
        # only two part 
        #sigmoid(drop_target - mean_drop_other) 
        # 3 2 =1
        # -3 -4 =1
        #res_score[df_tmp['method_name'][0]][tmp_taget_lang] = f((tmp_drop[tmp_taget_lang]) -((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2))
        #res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
    
        #sigmoid(drop_target) - sigmoid(mean_drop_other) 
        #res_score[df_tmp['method_name'][0]][tmp_taget_lang] = f(tmp_drop[tmp_taget_lang]) -f((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2)
        #res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
    
        '''
             precision = target_drop / (target_drop + max_other_drop + EPS)
                recall = target_drop / (org_model_task_result[target_task] + EPS)
                 f1 = 2 * precision * recall / (precision + recall + EPS)
    
                *_drop = max(org_drop, 0)#只考虑指标掉了的情况，如果指标增加了，说明没有找到暂时不考虑
    
        '''


        '''
            Alpha=0.7 sigmoid(alpha*exp(drop_target_rate) - (1-alpha)*exp(drop_other_rate))

        '''
        org_target = lang_score_baseline[tmp_taget_lang]
        target_drop = tmp_drop[tmp_taget_lang]
        #max_other_drop = max(tmp_drop[lang_list[0]], tmp_drop[lang_list[1]])

        # actually mean drop
        #max_other_drop = sum([tmp_drop[lang_list[0]], tmp_drop[lang_list[1]]])/2

        #actually sum
        max_other_drop = sum([tmp_drop[lang_list[0]], tmp_drop[lang_list[1]]])
        
        EPS =1e-12
        precision = target_drop / (target_drop + max_other_drop + EPS)
        recall = target_drop / (org_target + EPS)
        f1 = (1+ beta**2) * precision * recall / (precision + recall* beta**2 + EPS)
    
    
        res_score[df_tmp['method_name'][0]][tmp_taget_lang] = f1
        res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
        
    
        '''
            sigmoid(alpha*exp(drop_target) - (1-alpha)*exp(drop_other))
    
        '''
        if False:
            alpha=0.5 
            def sigmoid(x):
                return 1/(1 + np.exp(-x))
        
            drop_target = tmp_drop[tmp_taget_lang]
            drop_other_mean = (tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2
            
            res_score[df_tmp['method_name'][0]][tmp_taget_lang] = sigmoid(alpha*np.exp(drop_target) - (1- alpha)*np.exp(drop_other_mean))
            res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
        
            '''
                res_score[tmp_taget_lang] = alpha*(2**drop_target-1) - (1- alpha)*(2**abs(drop_other_mean)-1)
            '''
            alpha=0.8
            res_score[df_tmp['method_name'][0]][tmp_taget_lang] = alpha*(2**(drop_target)-1) - (1- alpha)*(2**abs(drop_other_mean)-1)
            res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
        
            
    
        if False:
            #LPS★ = (drop(target) - mean(drop(other))) * drop(target)
            drop_target = tmp_drop[tmp_taget_lang]
            mean_drop_other = ((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2)
            LPS_star = (drop_target - mean_drop_other) * drop_target
            res_score[df_tmp['method_name'][0]][tmp_taget_lang] = LPS_star

    return res_score, res_acuall_score, lang_score_baseline

def calc_metric_org(df_judge_res,result_dict, beta = 1 ):
    '''
        calc LPS:
            LPS = drop(target) / mean(drop(other)) * mean(drop)
    
        20251215 new metric:
            sigmoid(alpha*exp(drop_target) - (1-alpha)*exp(drop_other))
    
        20251217:
            增加一个版本:
                使用sigmoid(alpha*exp(drop_target_rate) - (1-alpha)*exp(drop_other_rate))计算NLU的一些指标
    
        20251218:
         precision = target_drop / (target_drop + max_other_drop + EPS)
                recall = target_drop / (org_model_task_result[target_task] + EPS)
                 f1 = 2 * precision * recall / (precision + recall + EPS)
    
                *_drop = max(org_drop, 0)#只考虑指标掉了的情况，如果指标增加了，说明没有找到暂时不考虑
    
    '''
    
    # get org
    df_judge_res['specific_method'] = df_judge_res['class_tpye'].apply(lambda x: x.split('|')[1])
    df_org = df_judge_res.loc[df_judge_res['specific_method'] =='org_model' ]
    
    # 获取不同语种测试集合的分数
    def get_all_lang_score(df_org):
        lang_list = ['en', 'vi', 'zh']
        lang_score_baseline = {}
        for ilang in lang_list:
            lang_score_baseline[ilang] = df_org.loc[df_org['lang']==ilang]['score'].mean()
        return lang_score_baseline
    # baseline
    lang_score_baseline = get_all_lang_score(df_org)
    
    # calc LPS:
    res_score = {}
    res_acuall_score = {}
    for iexp_name in result_dict.keys():
        specific_method = iexp_name.split('|')[1]
        if specific_method=='org_model':continue
    
        df_tmp = df_judge_res.loc[df_judge_res['specific_method']==specific_method ].reset_index(drop=True)
        lang_score_tmp = get_all_lang_score(df_tmp)
    
        assert len(dict(df_tmp['mask_neuron_lang'].value_counts()))==1 and len(dict(df_tmp['method_name'].value_counts()))==1
    
       
    
        assert df_tmp['mask_neuron_lang'][0] not in res_score,(specific_method, df_tmp['mask_neuron_lang'][0], res_score)
    
        if df_tmp['method_name'][0] not in res_score:
            res_score[df_tmp['method_name'][0]] = {}
            res_acuall_score[df_tmp['method_name'][0]] = {}
             
        # calc per lang score
        tmp_drop = {}
        for ikey in lang_score_baseline.keys():
            # drop
            #tmp_drop[ikey] = lang_score_baseline[ikey] - lang_score_tmp[ikey]
            # drop rate
            #tmp_drop[ikey] = (lang_score_baseline[ikey] - lang_score_tmp[ikey])/lang_score_baseline[ikey]
            
            tmp_drop[ikey] = max(lang_score_baseline[ikey] - lang_score_tmp[ikey], 0)
    
        tmp_taget_lang = df_tmp['mask_neuron_lang'][0]
        lang_list = ['en', 'vi', 'zh']
        lang_list.remove(tmp_taget_lang)
    
        # LPS
        #LPS = drop(target) / mean(drop(other)) * mean(drop)
        #LPS = tmp_drop[tmp_taget_lang]/((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2)*(sum(tmp_drop.values())/3)
        #res_score[df_tmp['method_name'][0]][tmp_taget_lang] = LPS
    
        # sigmoid LPS
        k = 2
        f = lambda x: 2/(1+np.exp(-k*x))
        # sigmoid(drop_target) /sigmoid(mean_drop_other) * sigmoid(all_mean_drop) 
        
        #LPS_sig = f(tmp_drop[tmp_taget_lang]) / f(((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2)) #* f(sum(tmp_drop.values())/3)
        
        #res_score[df_tmp['method_name'][0]][tmp_taget_lang] = LPS_sig
    
        #res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] = lang_score_tmp
    
    
        # only two part 
        #sigmoid(drop_target - mean_drop_other) 
        # 3 2 =1
        # -3 -4 =1
        #res_score[df_tmp['method_name'][0]][tmp_taget_lang] = f((tmp_drop[tmp_taget_lang]) -((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2))
        #res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
    
        #sigmoid(drop_target) - sigmoid(mean_drop_other) 
        #res_score[df_tmp['method_name'][0]][tmp_taget_lang] = f(tmp_drop[tmp_taget_lang]) -f((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2)
        #res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
    
        '''
             precision = target_drop / (target_drop + max_other_drop + EPS)
                recall = target_drop / (org_model_task_result[target_task] + EPS)
                 f1 = 2 * precision * recall / (precision + recall + EPS)
    
                *_drop = max(org_drop, 0)#只考虑指标掉了的情况，如果指标增加了，说明没有找到暂时不考虑
    
        '''
        org_target = lang_score_baseline[tmp_taget_lang]
        target_drop = tmp_drop[tmp_taget_lang]
        #max_other_drop = max(tmp_drop[lang_list[0]], tmp_drop[lang_list[1]])

        # actually mean drop
        max_other_drop = sum([tmp_drop[lang_list[0]], tmp_drop[lang_list[1]]])/2
        
        EPS =1e-12
        precision = target_drop / (target_drop + max_other_drop + EPS)
        recall = target_drop / (org_target + EPS)
        f1 = (1+ beta**2) * precision * recall / (precision + recall* beta**2 + EPS)
    
    
        res_score[df_tmp['method_name'][0]][tmp_taget_lang] = f1
        res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
        
    
        '''
            sigmoid(alpha*exp(drop_target) - (1-alpha)*exp(drop_other))
    
        '''
        if False:
            alpha=0.5 
            def sigmoid(x):
                return 1/(1 + np.exp(-x))
        
            drop_target = tmp_drop[tmp_taget_lang]
            drop_other_mean = (tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2
            
            res_score[df_tmp['method_name'][0]][tmp_taget_lang] = sigmoid(alpha*np.exp(drop_target) - (1- alpha)*np.exp(drop_other_mean))
            res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
        
            '''
                res_score[tmp_taget_lang] = alpha*(2**drop_target-1) - (1- alpha)*(2**abs(drop_other_mean)-1)
            '''
            alpha=0.8
            res_score[df_tmp['method_name'][0]][tmp_taget_lang] = alpha*(2**(drop_target)-1) - (1- alpha)*(2**abs(drop_other_mean)-1)
            res_acuall_score[df_tmp['method_name'][0]][tmp_taget_lang] =  lang_score_tmp
        
            
    
        if False:
            #LPS★ = (drop(target) - mean(drop(other))) * drop(target)
            drop_target = tmp_drop[tmp_taget_lang]
            mean_drop_other = ((tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2)
            LPS_star = (drop_target - mean_drop_other) * drop_target
            res_score[df_tmp['method_name'][0]][tmp_taget_lang] = LPS_star

    return res_score, res_acuall_score, lang_score_baseline



    
def show_result(mname,res_acuall_score, res_score,  digits=3):
    print("\n" + "=" * 60)
    print(f"Method: {mname}")
    print("=" * 60)

    data = res_acuall_score[mname]

    # ---- normalize to: list[(tgt, {eval: score})] ----
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, (list, tuple)):
        # could be list[(tgt, dict)] already
        items = list(data)
    else:
        raise TypeError(f"Unsupported type for res_acuall_score[{mname!r}]: {type(data)}")

    # infer eval langs (union + stable order)
    eval_langs = []
    seen = set()
    for _, d in items:
        for k in d.keys():
            if k not in seen:
                seen.add(k)
                eval_langs.append(k)

    # header
    print("{:<12}".format("Target\\Eval"), end="")
    for l in eval_langs:
        print("{:>10}".format(l), end="")
    print()
    print("-" * (12 + 10 * len(eval_langs)))

    # rows
    for tgt, d in sorted(items, key=lambda x: x[0]):
        print("{:<12}".format(tgt), end="")
        for l in eval_langs:
            v = d.get(l, None)
            if v is None:
                s = "NA"
            else:
                s = f"{float(v):.{digits}f}"
            print("{:>10}".format(s), end="")
        print()

    # langspec
    print("\nLangSpec-F1:")
    for k, v in sorted(res_score[mname].items(), key=lambda x: x[0]):
        print(f"  {k:<4}: {float(v):.{digits}f}")

    print("=" * 60)



def get_nlu_metric_score(lang_score_baseline, lang_score_tmp, method_name, task_list, target_task, beta=1):

    
    res_score = {}
    res_acuall_score = {}
    # calc drop
   
    # calc per lang score
    tmp_drop = {}
    for i_taskname in lang_score_baseline.keys():
        # drop value
        #tmp_drop[i_taskname] = lang_score_baseline[i_taskname] - lang_score_tmp[i_taskname]
        # drop rate
        #tmp_drop[i_taskname] = (lang_score_baseline[i_taskname] - lang_score_tmp[i_taskname])/lang_score_baseline[i_taskname]

        # just drop case
        tmp_drop[i_taskname] = max(lang_score_baseline[i_taskname] - lang_score_tmp[i_taskname], 0 )
    

    tmp_taget_lang = target_task
    lang_list = task_list
    lang_list.remove(tmp_taget_lang)


    alpha=0.7
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    org_target = lang_score_baseline[tmp_taget_lang]
    drop_target = tmp_drop[tmp_taget_lang]
    drop_other_mean = (tmp_drop[lang_list[0]] + tmp_drop[lang_list[1]])/2

    
    #res_score[tmp_taget_lang] = sigmoid(alpha*np.exp(1*drop_target) - (1- alpha)*np.exp(1*(drop_other_mean)))

    #score = 2**(drop_rate)- 2**abs(other_drop_rate)
    #res_score[tmp_taget_lang] = (alpha*2**drop_target - (1- alpha)*2**abs(drop_other_mean))/(3*alpha - 1)
    
    #score = ((alpha)*(2**(drop_rate)-1)- (1-alpha)(2**abs(other_drop_rate)-1)
    #res_score[tmp_taget_lang] = alpha*(2**drop_target-1) - (1- alpha)*(2**abs(drop_other_mean)-1)

    #alpha = 0.8 beta = 6 score = sigmoid(beta*((alpha)*(2**(drop_rate)-1)- (1-alpha)(2**abs(other_drop_rate)-1)))
    #beta = 6
    #res_score[tmp_taget_lang] = sigmoid(beta*(alpha*(2**drop_target-1) - (1- alpha)*(2**abs(drop_other_mean)-1) ))


    '''
            precision = target_drop / (target_drop + max_other_drop + EPS)
            recall = target_drop / (org_model_task_result[target_task] + EPS)
            f1 = 2 * precision * recall / (precision + recall + EPS)

            *_drop = max(org_drop, 0)#只考虑指标掉了的情况，如果指标增加了，说明没有找到暂时不考虑
    '''
    # langspec-F1_v1
    if False:
        EPS=1e-12
        precision = drop_target/(drop_target + max(tmp_drop[lang_list[0]], tmp_drop[lang_list[1]]) + EPS)
        recall = drop_target/(org_target + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)

    #langspec-F1_v2 
    EPS=1e-12
    precision = drop_target/(drop_target +sum([tmp_drop[lang_list[0]], tmp_drop[lang_list[1]]])/2 + EPS)
    recall = drop_target/(org_target + EPS)
    f1 = (1+ beta**2) * precision * recall / (precision + recall* beta**2 + EPS)

    res_score[tmp_taget_lang] =  f1

    return res_score


    
    

    
    

    

    
































    
