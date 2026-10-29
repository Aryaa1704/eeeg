from __future__ import annotations
from pathlib import Path
import json, platform
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from .config import ExperimentConfig, CLASS_NAMES
from .data import load_gdf, find_subject_sessions
from .model import filter_trials, crop, make_model
from .stats import mcnemar_exact, group_wilcoxon, holm

def _score(train_x, train_y, test_x, test_y, cfg):
    model=make_model(cfg.csp_components); model.fit(train_x, train_y); pred=model.predict(test_x)
    return pred, float(accuracy_score(test_y,pred))

def choose_window(train_x, y, cfg):
    """Training-session CV selection only; no held-out session scores are consulted."""
    candidates=[]; cv=StratifiedKFold(3,shuffle=True,random_state=cfg.random_seed)
    for width in cfg.search_widths:
        for start in np.arange(cfg.search_start_range[0], cfg.search_start_range[1]+1e-9, cfg.step_seconds):
            scores=[]; w=(round(float(start),2),round(float(start+width),2)); xx=crop(train_x,cfg.sfreq,w)
            for tr,va in cv.split(xx,y): scores.append(_score(xx[tr],y[tr],xx[va],y[va],cfg)[1])
            candidates.append((w,float(np.mean(scores))))
    best_window, best_score = max(candidates, key=lambda x: x[1])
    return best_window, best_score, candidates

def run(data_root: Path, output: Path, cfg: ExperimentConfig, exploratory_test_window_search=False) -> None:
    output.mkdir(parents=True,exist_ok=True); (output/"confusion_matrices").mkdir(exist_ok=True); (output/"figures").mkdir(exist_ok=True)
    rows=[]; metrics=[]; windows=[]; temporal_rows=[]; tests=[]
    for subject in range(1,10):
        train_path,test_path,label_path=find_subject_sessions(data_root,subject)
        if not train_path.exists() or not test_path.exists() or not label_path.exists(): continue
        train=load_gdf(train_path); test=load_gdf(test_path, label_path)
        if not np.array_equal(train.sfreq,test.sfreq): raise ValueError(f"Subject {subject}: sample rates differ")
        cfg=ExperimentConfig(**(cfg.__dict__ | {"sfreq":train.sfreq}))
        tx=filter_trials(train.data,train.sfreq,cfg.band_limits(),cfg.filter_order); ex=filter_trials(test.data,test.sfreq,cfg.band_limits(),cfg.filter_order)
        # Conditions share the same held-out MI test window; only training coverage differs.
        conditions={"motor_imagery_only":cfg.mi_window,"visual_cue_inclusive":cfg.cue_window}
        predictions={}
        for condition, train_window in conditions.items():
            pred,acc=_score(crop(tx,train.sfreq,train_window),train.labels,crop(ex,test.sfreq,cfg.mi_window),test.labels,cfg); predictions[condition]=pred
            rows.append({"subject":subject,"condition":condition,"accuracy":acc,"train_interval_s":train_window,"test_interval_s":cfg.mi_window,"band_hz":cfg.band_limits(),"classifier":"CSP+LDA","split":"day1_train/day2_test"})
            rep=classification_report(test.labels,pred,labels=range(4),target_names=CLASS_NAMES,output_dict=True,zero_division=0)
            for name in CLASS_NAMES: metrics.append({"subject":subject,"condition":condition,"class":name,**rep[name]})
            cm=confusion_matrix(test.labels,pred,labels=range(4)); np.save(output/"confusion_matrices"/f"subject_{subject:02d}_{condition}.npy",cm)
            (output/"confusion_matrices"/f"subject_{subject:02d}_{condition}.json").write_text(json.dumps({"labels":CLASS_NAMES,"matrix":cm.tolist(),"kappa":cohen_kappa_score(test.labels,pred)},indent=2))
        baseline=rows[-2]["accuracy"]; visual=rows[-1]["accuracy"]; rows[-1]["difference_pp_vs_mi"]=100*(visual-baseline); rows[-2]["difference_pp_vs_mi"]=0.
        tests.append({"subject":subject,**mcnemar_exact(predictions['motor_imagery_only']==test.labels,predictions['visual_cue_inclusive']==test.labels),"null_hypothesis":"Equal paired trial correctness"})
        selected, cv_score, candidates = choose_window(tx,train.labels,cfg)
        for window, score in candidates:
            temporal_rows.append({"subject": subject, "window_start_s": window[0], "window_end_s": window[1],
                                  "width_s": window[1] - window[0], "training_cv_accuracy": score,
                                  "selection": "training_CV", "exploratory_test_scored": False,
                                  "band_hz": cfg.band_limits(), "classifier": "CSP4+OVO-LDA"})
        pred,acc=_score(crop(tx,train.sfreq,selected),train.labels,crop(ex,test.sfreq,cfg.mi_window),test.labels,cfg)
        windows.append({"subject":subject,"selection":"training_CV","window_start_s":selected[0],"window_end_s":selected[1],"width_s":selected[1]-selected[0],"cv_accuracy":cv_score,"heldout_test_accuracy":acc,"band_hz":cfg.band_limits()})
    acc=pd.DataFrame(rows); acc.to_csv(output/"subject_wise_accuracy.csv",index=False); pd.DataFrame(metrics).to_csv(output/"per_class_metrics.csv",index=False); pd.DataFrame(windows).to_csv(output/"optimal_windows.csv",index=False)
    pd.DataFrame(temporal_rows).to_csv(output/"temporal_window_results.csv",index=False)
    if tests:
        adjusted = holm([test["p_value"] for test in tests])
        for test, adjusted_p in zip(tests, adjusted): test["adjusted_p_value_holm"] = adjusted_p
        group=group_wilcoxon(acc[acc.condition=='visual_cue_inclusive'].accuracy.to_numpy()-acc[acc.condition=='motor_imagery_only'].accuracy.to_numpy())
        tests.append({"subject":"group",**group,"null_hypothesis":"Median paired subject accuracy difference is zero","adjusted_p_value_holm":None})
    pd.DataFrame(tests).to_csv(output/"statistical_tests.csv",index=False)
    summary={"subjects_completed":int(acc.subject.nunique()) if len(acc) else 0,"status":"completed" if len(acc) else "no labelled session pairs found; no results fabricated","config":cfg.metadata(),"python":platform.python_version()}
    (output/"summary_statistics.json").write_text(json.dumps(summary,indent=2)); (output/"experiment_config.json").write_text(json.dumps(cfg.metadata(),indent=2))
