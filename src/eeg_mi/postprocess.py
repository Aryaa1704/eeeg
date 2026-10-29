"""Combine completed strict experiments into auditable statistics and figures."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def load(root: Path, band: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accuracy = pd.read_csv(root / "subject_wise_accuracy.csv"); accuracy["band"] = band
    temporal = pd.read_csv(root / "temporal_window_results.csv"); temporal["band"] = band
    metrics = pd.read_csv(root / "per_class_metrics.csv"); metrics["band"] = band
    return accuracy, temporal, metrics

def run(paper: Path, band8: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True); figures = output / "figures"; figures.mkdir(exist_ok=True)
    a1,t1,m1=load(paper,"0.5-30 Hz"); a2,t2,m2=load(band8,"8-30 Hz")
    accuracy=pd.concat([a1,a2],ignore_index=True); temporal=pd.concat([t1,t2],ignore_index=True); metrics=pd.concat([m1,m2],ignore_index=True)
    accuracy.to_csv(output/"combined_subject_wise_accuracy.csv",index=False); temporal.to_csv(output/"combined_temporal_window_results.csv",index=False); metrics.to_csv(output/"combined_per_class_metrics.csv",index=False)
    # Paired visual-training contrast, calculated from actual held-out predictions.
    rows=[]
    for band, frame in accuracy.groupby("band"):
        wide=frame.pivot(index="subject",columns="condition",values="accuracy")
        delta=wide["visual_cue_inclusive"]-wide["motor_imagery_only"]
        stat,p=wilcoxon(delta) if not (delta==0).all() else (0.,1.)
        rows.append({"band":band,"comparison":"visual_cue_inclusive minus motor_imagery_only","n_subjects":len(delta),"mean_difference_pp":100*delta.mean(),"median_difference_pp":100*delta.median(),"wilcoxon_statistic":float(stat),"p_value":float(p)})
    pd.DataFrame(rows).to_csv(output/"combined_group_statistics.csv",index=False)
    # Subject accuracy grouped bars.
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,(band,frame) in zip(axes,accuracy.groupby("band")):
        wide=frame.pivot(index="subject",columns="condition",values="accuracy")*100
        wide.plot.bar(ax=ax,color=["#33658A","#F26419"],width=.8)
        ax.set(title=band,xlabel="Subject",ylabel="Held-out accuracy (%)",ylim=(0,100)); ax.legend(title="Training condition")
    fig.tight_layout(); fig.savefig(figures/"subject_wise_accuracy_comparison.png",dpi=220); plt.close(fig)
    # Mean training-CV accuracy against window start.
    fig,ax=plt.subplots(figsize=(8,5))
    for band,frame in temporal.groupby("band"):
        curve=frame.groupby("window_start_s").training_cv_accuracy.agg(["mean","std"])
        ax.plot(curve.index,curve["mean"]*100,marker="o",label=band); ax.fill_between(curve.index,(curve["mean"]-curve["std"])*100,(curve["mean"]+curve["std"])*100,alpha=.15)
    ax.axvline(3.,color="black",linestyle="--",label="MI baseline start"); ax.set(xlabel="Training-window start (s)",ylabel="Training-CV accuracy (%)",title="Accuracy by temporal window"); ax.legend(); fig.tight_layout(); fig.savefig(figures/"accuracy_vs_window_start.png",dpi=220); plt.close(fig)
    # Subject-by-window heatmaps for each frequency band.
    for band,frame in temporal.groupby("band"):
        matrix=frame.pivot(index="subject",columns="window_start_s",values="training_cv_accuracy")*100
        fig,ax=plt.subplots(figsize=(9,5)); image=ax.imshow(matrix,aspect="auto",cmap="viridis",vmin=25,vmax=100)
        ax.set(yticks=range(len(matrix.index)),yticklabels=[f"S{x:02d}" for x in matrix.index],xticks=range(len(matrix.columns)),xticklabels=matrix.columns,xlabel="Window start (s)",ylabel="Subject",title=f"Training-CV accuracy heatmap ({band})")
        fig.colorbar(image,ax=ax,label="Accuracy (%)"); fig.tight_layout(); fig.savefig(figures/f"heatmap_{band.replace('.','').replace(' ','_')}.png",dpi=220); plt.close(fig)
    # Per-class F1 comparison.
    summary=metrics.groupby(["band","condition","class"],as_index=False)["f1-score"].mean()
    fig,ax=plt.subplots(figsize=(10,5)); labels=[]; values=[]
    for band in summary.band.unique():
        for condition in ["motor_imagery_only","visual_cue_inclusive"]:
            sub=summary[(summary.band==band)&(summary.condition==condition)]; labels += [f"{band}\n{condition}\n{x}" for x in sub["class"]]; values += (sub["f1-score"]*100).tolist()
    ax.bar(range(len(values)),values,color="#33658A"); ax.set(xticks=range(len(values)),xticklabels=labels,ylabel="Mean F1 (%)",title="Per-class F1 across conditions and bands",ylim=(0,100)); ax.tick_params(axis="x",labelrotation=90); fig.tight_layout(); fig.savefig(figures/"per_class_f1_comparison.png",dpi=220); plt.close(fig)
    (output/"postprocess_manifest.json").write_text(json.dumps({"source_results":[str(paper),str(band8)],"figures":[p.name for p in figures.glob("*.png")]},indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("paper",type=Path); p.add_argument("band8",type=Path); p.add_argument("--output",type=Path,default=Path("results_final")); args=p.parse_args(); run(args.paper,args.band8,args.output)
