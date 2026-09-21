from pathlib import Path
import pandas as pd

def test_primary_regression():
    root=Path(__file__).resolve().parents[1]; p=root/'RESULTS'/'primary_gmv'/'gmv_summary.csv'
    assert p.exists(); s=pd.read_csv(p).set_index('method')
    exp={'AIQ':4.1235,'LS3':4.16044,'ID_KAPPA':4.16944,'HC':4.17137,'GS_STAR':4.33476,'SRE_STAR':4.33020}
    for k,v in exp.items(): assert abs(float(s.loc[k,'realised_vol_10bp_pct'])-v)<0.003
    assert 'AIQSG_CHI1' not in s.index

def test_final_reported_metrics_and_exclusions():
    root=Path(__file__).resolve().parents[1]; s=pd.read_csv(root/'RESULTS'/'primary_gmv'/'gmv_summary.csv')
    required={'realised_vol_10bp_pct','sharpe_10bp','max_drawdown_10bp_pct','es95_10bp_pct','downside_deviation_10bp_pct','terminal_wealth_10bp_from_100'}
    assert required.issubset(s.columns)
    assert {'terminal_wealth_0bp_from_100','gross_turnover_ann'}.isdisjoint(s.columns)

def test_turnover_retained_only_in_monthly_accounting():
    root=Path(__file__).resolve().parents[1]; p=pd.read_csv(root/'RESULTS'/'primary_gmv'/'monthly_gmv_results.csv')
    assert 'gross_turnover' in p.columns and 'gross_return' in p.columns and 'net_return' in p.columns

def test_atomic_terminal_wealth_and_tail_metrics():
    root=Path(__file__).resolve().parents[1]; a=pd.read_csv(root/'RESULTS'/'primary_gmv'/'gmv_summary.csv').set_index('method').loc['AIQ']
    assert abs(float(a['terminal_wealth_10bp_from_100'])-255.05)<0.10
    assert abs(float(a['es95_10bp_pct'])-2.791)<0.01
    assert abs(float(a['downside_deviation_10bp_pct'])-2.633)<0.01

def test_ranking_panels():
    root=Path(__file__).resolve().parents[1]
    risk=pd.read_csv(root/'RESULTS'/'primary_gmv'/'gmv_risk_ranking.csv').set_index('method')
    broad=pd.read_csv(root/'RESULTS'/'primary_gmv'/'gmv_broad_ranking.csv').set_index('method')
    assert int(risk.loc['AIQ','risk_aggregate'])==4 and int(risk.loc['AIQ','risk_overall_rank'])==1
    assert int(broad.loc['AIQ','broad_aggregate'])==30 and int(broad.loc['AIQ','broad_overall_rank'])==1

def test_publication_table_has_one_atomic_row():
    root=Path(__file__).resolve().parents[1]; t=pd.read_csv(root/'RESULTS'/'primary_gmv'/'gmv_publication_table.csv')
    assert (t['method']=='AIQ').sum()==1
    assert not t['method'].astype(str).str.contains('CHI1|AIQSG',case=False,regex=True).any()
