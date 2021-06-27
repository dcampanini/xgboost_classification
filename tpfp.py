#%%
print('tp=',701, 'fp=', 188, 'tn=', 1074, 'fn=',2949)
#%%
thr=0.99
#
tp=486
fp=3327
tn=3123
fn=303
#%% calcular precision y recall
precision=tp/(tp+fp)
recall=tp/(tp+fn)
print('precision:',precision*100)
print('recall:',recall*100)
# %%
total=tp+fp+tn+fn
pos=(tp+fn)
neg=(tn+fp)
print('numero pos:',pos)
print('num negativos:',neg)
print('% pos:',pos*100/total)
print('% neg',neg*100/total)
# %%
accuracy=(tp+tn)/total
print('accuracy:',accuracy*100)

# %%
