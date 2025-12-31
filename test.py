import pulp

def test_ai_factory_model():
    lp=pulp.LpProblem("AI_Factory_Test",pulp.LpMinimize)

    # Variables
    x_BERT_A100=pulp.LpVariable('x_BERT_A100',lowBound=0)
    x_BERT_V100=pulp.LpVariable('x_BERT_V100', lowBound=0)
    x_BERT_T4=pulp.LpVariable('x_BERT_T4',lowBound=0)

    # Fonction objectif
    lp+=20*x_BERT_A100+35*x_BERT_V100+80*x_BERT_T4

    # Contrainte de commande
    lp+=x_BERT_A100+x_BERT_V100+x_BERT_T4>=30

    # Résolution
    lp.solve()

    # Tests unitaires
    assert pulp.LpStatus[lp.status]=="Optimal"
    assert pulp.value(lp.objective)>=0
    assert x_BERT_A100.varValue+x_BERT_V100.varValue+x_BERT_T4.varValue>=30
