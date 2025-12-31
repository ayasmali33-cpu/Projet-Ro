import pulp

# -------------------------
#Creation du modele ou probleme lineaire
# -------------------------
lp=pulp.LpProblem("AI_Factory_Optimisation",pulp.LpMinimize)

# -------------------------
#Variables de decisions
# -------------------------
x_BERT_A100=pulp.LpVariable('x_BERT_A100',lowBound=0,cat='Continuous')
x_BERT_V100=pulp.LpVariable('x_BERT_V100',lowBound=0,cat='Continuous')
x_BERT_T4=pulp.LpVariable('x_BERT_T4',lowBound=0,cat='Continuous')

x_ResNet_A100=pulp.LpVariable('x_ResNet_A100',lowBound=0, cat='Continuous')
x_ResNet_V100=pulp.LpVariable('x_ResNet_V100',lowBound=0, cat='Continuous')
x_ResNet_T4=pulp.LpVariable('x_ResNet_T4',lowBound=0, cat='Continuous')

x_LSTM_A100=pulp.LpVariable('x_LSTM_A100',lowBound=0,cat='Continuous')
x_LSTM_V100=pulp.LpVariable('x_LSTM_V100',lowBound=0,cat='Continuous')
x_LSTM_T4=pulp.LpVariable('x_LSTM_T4',lowBound=0,cat='Continuous')

# -------------------------
#Fonction objectif
# -------------------------
lp+=(
    20*x_BERT_A100+35*x_BERT_V100+80*x_BERT_T4
  +15*x_ResNet_A100+20*x_ResNet_V100+40*x_ResNet_T4
  +5*x_LSTM_A100+8*x_LSTM_V100+12*x_LSTM_T4
)

#----LES 10 CONTRAINTES DEMANDEES
# -------------------------
#Contraintes:commandes clients
# -------------------------
lp+=x_BERT_A100+x_BERT_V100+x_BERT_T4>=30
lp+=x_ResNet_A100+x_ResNet_V100+x_ResNet_T4>=50
lp+=x_LSTM_A100+x_LSTM_V100+x_LSTM_T4>=40

# -------------------------
#Contraintes:disponibilite des GPU (heures)
#Hypothese:temps proportionnel au cout
# -------------------------
lp+=x_BERT_A100+x_ResNet_A100+x_LSTM_A100<=200
lp+=2*x_BERT_V100+2*x_ResNet_V100+2*x_LSTM_V100<=300
lp+=4*x_BERT_T4+4*x_ResNet_T4+4*x_LSTM_T4<=500

# -------------------------
#Contraintes:preference pour le GPU A100
# -------------------------
lp+=x_BERT_A100>=x_BERT_V100+x_BERT_T4
lp+=x_ResNet_A100>=x_ResNet_V100+x_ResNet_T4
lp+=x_LSTM_A100>=x_LSTM_V100+x_LSTM_T4

# -------------------------
#Resolution
# -------------------------
lp.solve()

# -------------------------
#les resultats obtenues
# -------------------------
print("Statut de la solution:",pulp.LpStatus[lp.status])
print("Cout total minimal=",pulp.value(lp.objective))
for v in lp.variables():
    print(v.name,"=",v.varValue)
# -------------------------
#Analyse des goulots d'etranglement GPU
# -------------------------
#Heures utilisees par chaque type de GPU
used_A100=x_BERT_A100.varValue+x_ResNet_A100.varValue+x_LSTM_A100.varValue
used_V100=2*x_BERT_V100.varValue+2*x_ResNet_V100.varValue+2*x_LSTM_V100.varValue
used_T4=4*x_BERT_T4.varValue+4*x_ResNet_T4.varValue+4*x_LSTM_T4.varValue

print("Analyse des goulots d'etranglement\n")
print("Heures utilisees GPU A100 :",used_A100,"/ 200")
print("Heures utilisees GPU V100 :",used_V100,"/ 300")
print("Heures utilisees GPU T4 :",used_T4,"/ 500")

if used_A100>=200:
    print("GPU A100 est un GOULOT D'ETRANGLEMENT")
if used_V100>=300:
    print("GPU V100 est un GOULOT D'ETRANGLEMENT")
if used_T4>=500:
    print("GPU T4 est un GOULOT D'ETRANGLEMENT")

