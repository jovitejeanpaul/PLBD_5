import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import pickle

# ── Modèle LSTM ───────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=5, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size * horizon)
        self.horizon = horizon
        self.output_size = output_size

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.horizon, self.output_size)

# ── Chargement ────────────────────────────────────────────────────────────────
model = LSTMModel()
model.load_state_dict(torch.load('models/lstm_model.pth'))
model.eval()

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

FEATURES = ['pH', 'TDS', 'Turbidité', 'Conductivité', 'Température']
UNITS    = ['', 'mg/L', 'NTU', 'µS/cm', '°C']
COLORS   = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

OMS = {
    'pH':          (6.5, 8.5),
    'TDS':         (50,  500),
    'Turbidité':   (0,   4),
    'Conductivité':(200, 800),
    'Température': (10,  30),
}

LIMITES_PHYSIQUES = {
    'pH':          (0,   14),
    'TDS':         (0,   1000),
    'Turbidité':   (0,   100),
    'Conductivité':(0,   2000),
    'Température': (0,   50),
}

history = np.array([
    [7.20, 300.0, 2.20, 600.0, 21.0],
    [7.19, 301.0, 2.18, 602.0, 21.2],
    [7.21, 299.0, 2.22, 598.0, 21.5],
    [7.18, 302.0, 2.19, 604.0, 21.8],
    [7.20, 300.5, 2.21, 601.0, 22.0],
    [7.22, 299.5, 2.17, 599.0, 22.3],
    [7.19, 301.5, 2.23, 603.0, 22.5],
    [7.21, 300.0, 2.20, 600.0, 22.8],
    [7.20, 300.5, 2.19, 601.0, 23.0],
    [7.19, 301.0, 2.21, 602.0, 23.2],
    [7.21, 299.5, 2.18, 599.0, 23.5],
    [7.20, 300.0, 2.22, 600.0, 23.8],
    [7.22, 301.0, 2.20, 602.0, 24.0],
    [7.19, 300.5, 2.19, 601.0, 24.2],
    [7.21, 299.0, 2.21, 598.0, 24.5],
    [7.20, 301.5, 2.23, 603.0, 24.3],
    [7.19, 300.0, 2.20, 600.0, 24.0],
    [7.21, 300.5, 2.18, 601.0, 23.8],
    [7.20, 301.0, 2.22, 602.0, 23.5],
    [7.22, 299.5, 2.19, 599.0, 23.2],
    [7.19, 300.0, 2.21, 600.0, 23.0],
    [7.21, 301.0, 2.20, 602.0, 22.8],
    [7.20, 300.5, 2.22, 601.0, 22.5],
    [7.19, 300.0, 2.19, 600.0, 22.2],
])
def predict(sequence):
    seq_scaled = scaler.transform(sequence)
    tensor = torch.FloatTensor(seq_scaled[-10:]).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(tensor).numpy()[0]
    shape = pred_scaled.shape
    return scaler.inverse_transform(pred_scaled.reshape(-1, 5)).reshape(shape)

def add_oms_zones(ax, feat, y_min, y_max):
    lo, hi = OMS[feat]
    ax.axhspan(lo, hi, alpha=0.08, color='green')
    if y_min < lo:
        ax.axhspan(y_min, lo, alpha=0.08, color='red')
    if y_max > hi:
        ax.axhspan(hi, y_max, alpha=0.08, color='red')

def analyser_tendance(pred):
    lignes = []
    conclusion_score = 0
    alerte_forte = False

    for i, feat in enumerate(FEATURES):
        valeurs = pred[:, i]
        lo, hi = OMS[feat]
        debut = valeurs[:4].mean()
        fin = valeurs[20:].mean()
        variation = ((fin - debut) / abs(debut)) * 100 if debut != 0 else 0
        hors_norme = valeurs.max() > hi or valeurs.min() < lo
        pct_hors = np.mean((valeurs > hi) | (valeurs < lo)) * 100
        diffs = np.abs(np.diff(valeurs))
        variation_brusque = diffs.max() / abs(debut) * 100 if debut != 0 else 0

        if hors_norme and pct_hors > 30:
            icone = "🔴"
            conclusion_score += 2
            alerte_forte = True
        elif hors_norme or abs(variation) > 15:
            icone = "⚠️"
            conclusion_score += 1
        elif abs(variation) > 5:
            icone = "🟡"
            conclusion_score += 0.5
        else:
            icone = "✅"

        if variation > 5:
            direction = f"↗ En hausse ({variation:.1f}%)"
        elif variation < -5:
            direction = f"↘ En baisse ({abs(variation):.1f}%)"
        else:
            direction = "→ Stable"

        if feat == 'Turbidité' and fin > 4:
            detail = "→ Possible apport de matières en suspension"
        elif feat == 'Turbidité' and variation > 20:
            detail = "→ Hausse progressive, surveiller source de contamination"
        elif feat == 'pH' and fin < 6.5:
            detail = "→ Acidification détectée, possible contamination chimique"
        elif feat == 'pH' and fin > 8.5:
            detail = "→ Basicité excessive, vérifier source d'alcalinité"
        elif feat == 'Température' and fin > 25:
            detail = "→ Température élevée, risque de prolifération bactérienne"
        elif feat == 'TDS' and fin > 500:
            detail = "→ Minéralisation excessive, eau non recommandée"
        elif feat == 'Conductivité' and variation > 15:
            detail = "→ Hausse ionique, possible pollution"
        elif variation_brusque > 20:
            detail = "→ Variation brusque détectée, événement externe possible"
        else:
            detail = ""

        unit = UNITS[i]
        label = f"{feat} ({unit})" if unit else feat
        ligne = f"{icone}  {label:<22} {direction}"
        if detail:
            ligne += f"\n      {detail}"
        lignes.append(ligne)

    lignes.append("")
    if alerte_forte or conclusion_score >= 3:
        lignes.append("🔴  CONCLUSION : Dégradation significative prévue")
        lignes.append("     Intervention préventive recommandée dans les 6h")
        lignes.append("     Source : OMS Guidelines for Drinking Water Quality 2022")
    elif conclusion_score >= 1.5:
        lignes.append("🟠  CONCLUSION : Dégradation modérée possible")
        lignes.append("     Surveillance renforcée recommandée")
        lignes.append("     Source : OMS Guidelines for Drinking Water Quality 2022")
    elif conclusion_score >= 0.5:
        lignes.append("🟡  CONCLUSION : Légères variations détectées")
        lignes.append("     Qualité globalement stable, surveillance normale")
    else:
        lignes.append("✅  CONCLUSION : Qualité de l'eau stable sur 24h")
        lignes.append("     Aucune anomalie prévue selon les normes OMS")

    return "\n".join(lignes)

def valider_sequence(entries_list, noms_lignes):
    valeurs = []
    erreurs = []
    avertissements = []
    features_saisies = ['pH', 'TDS', 'Turbidité', 'Température']

    for i, row in enumerate(entries_list):
        row_vals = []
        for j, feat in enumerate(features_saisies):
            try:
                val = float(row[j].get())
            except ValueError:
                erreurs.append(f"Ligne {noms_lignes[i]} — {feat} : valeur non numérique")
                row_vals.append(0)
                continue

            lo_phys, hi_phys = LIMITES_PHYSIQUES[feat]
            if not (lo_phys <= val <= hi_phys):
                erreurs.append(
                    f"Ligne {noms_lignes[i]} — {feat} = {val} : impossible !\n"
                    f"  Plage physique : {lo_phys} — {hi_phys}")
            else:
                lo_oms, hi_oms = OMS[feat]
                if not (lo_oms <= val <= hi_oms):
                    avertissements.append(
                        f"⚠️ Ligne {noms_lignes[i]} — {feat} = {val} hors norme OMS ({lo_oms}—{hi_oms})")
            row_vals.append(val)
        valeurs.append(row_vals)

    return valeurs, erreurs, avertissements

# ── Interface ─────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Aqua-Sens — Prévision Temporelle")
root.geometry("1200x750")
root.configure(bg='#1e1e2e')

tk.Label(root, text="💧 Aqua-Sens — Prévision de Qualité d'Eau",
         font=('Helvetica', 16, 'bold'), bg='#1e1e2e', fg='white').pack(pady=8)

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=10, pady=3)

style = ttk.Style()
style.theme_use('clam')
style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
style.configure('TNotebook.Tab', background='#313244', foreground='white',
                padding=[15, 5], font=('Helvetica', 10, 'bold'))
style.map('TNotebook.Tab', background=[('selected', '#89b4fa')])

# ════════════════════════════════════════════════════════
# ONGLET 1 — MODE AUTOMATIQUE
# ════════════════════════════════════════════════════════
tab1 = tk.Frame(notebook, bg='#1e1e2e')
notebook.add(tab1, text='📊 Mode Automatique')

pred_auto = predict(history)
analyse_auto = analyser_tendance(pred_auto)
now = datetime.now()

top1 = tk.Frame(tab1, bg='#1e1e2e')
top1.pack(fill='both', expand=True)

fig1, axes1 = plt.subplots(1, 5, figsize=(14, 3.5))
fig1.patch.set_facecolor('#1e1e2e')

for i, (ax, feat, color, unit) in enumerate(zip(axes1, FEATURES, COLORS, UNITS)):
    ax.set_facecolor('#313244')
    x_hist = np.arange(-24, 0)
    x_pred = np.arange(0, 24)
    all_vals = np.concatenate([history[:, i], pred_auto[:, i]])
    y_min, y_max = all_vals.min() * 0.98, all_vals.max() * 1.02
    add_oms_zones(ax, feat, y_min, y_max)
    ax.plot(x_hist, history[:, i], color=color, linewidth=1.5, label='Historique')
    ax.plot(x_pred, pred_auto[:, i], color='white', linewidth=1.5,
            linestyle='--', label='Prévision')
    ax.axvline(0, color='yellow', linewidth=1.5, linestyle=':', label='Maintenant')
    title = f"{feat}\n({unit})" if unit else feat
    ax.set_title(title, color='white', fontsize=9, fontweight='bold')
    ax.set_ylim(y_min, y_max)
    ax.tick_params(colors='white', labelsize=6)
    ax.set_ylabel(unit, color='#aaaaaa', fontsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')
    if i == 0:
        ax.legend(fontsize=6, facecolor='#313244', labelcolor='white')

fig1.suptitle(f"Prévision générée le {now.strftime('%d/%m/%Y à %Hh%M')}",
              color='#aaaaaa', fontsize=8)
fig1.tight_layout(pad=1.5)

canvas1 = FigureCanvasTkAgg(fig1, master=top1)
canvas1.draw()
canvas1.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)

bottom1 = tk.Frame(tab1, bg='#252535')
bottom1.pack(fill='x', padx=10, pady=3)
tk.Label(bottom1, text="🔍 ANALYSE DES TENDANCES — 24h à venir",
         font=('Helvetica', 10, 'bold'), bg='#252535', fg='#89b4fa').pack(anchor='w', padx=10, pady=3)
tk.Label(bottom1, text=analyse_auto, font=('Courier', 9), bg='#252535', fg='white',
         justify='left').pack(anchor='w', padx=15, pady=3)

# ════════════════════════════════════════════════════════
# ONGLET 2 — MODE MANUEL
# ════════════════════════════════════════════════════════
tab2 = tk.Frame(notebook, bg='#1e1e2e')
notebook.add(tab2, text='🎛️ Mode Manuel')

default_values = [
    [7.20, 300.0, 2.1, 20.0],
    [7.18, 301.0, 2.3, 20.2],
    [7.22, 299.0, 2.0, 20.4],
    [7.19, 302.0, 2.4, 20.6],
    [7.21, 300.5, 2.2, 20.8],
    [7.20, 301.5, 2.1, 21.0],
    [7.23, 298.0, 2.5, 21.2],
    [7.18, 303.0, 2.0, 21.4],
    [7.21, 300.0, 2.3, 21.6],
    [7.20, 300.0, 2.2, 21.8],
]

left = tk.Frame(tab2, bg='#1e1e2e')
left.pack(side='left', fill='y', padx=15, pady=10)

tk.Label(left, text="Entrez les 10 mesures (toutes modifiables) :",
         font=('Helvetica', 10, 'bold'), bg='#1e1e2e', fg='white').grid(
         row=0, column=0, columnspan=5, pady=8)

tk.Label(left, text="ℹ️ Conductivité calculée automatiquement (TDS × 2)",
         font=('Helvetica', 8, 'italic'), bg='#1e1e2e', fg='#89b4fa').grid(
         row=1, column=0, columnspan=5, pady=2)

headers = ['#', 'pH', 'TDS (mg/L)', 'Turb. (NTU)', 'Temp. (°C)']
for j, h in enumerate(headers):
    tk.Label(left, text=h, font=('Helvetica', 8, 'bold'),
             bg='#1e1e2e', fg='#89b4fa', width=12).grid(row=2, column=j, padx=3)

entries = []
for i in range(10):
    row_entries = []
    couleur_bg = '#2a2a3e' if i == 9 else '#313244'
    couleur_fg = 'yellow' if i == 9 else 'white'
    label_txt = f"Mesure {i+1} ►" if i == 9 else f"Mesure {i+1}"
    label_couleur = 'yellow' if i == 9 else '#aaaaaa'

    tk.Label(left, text=label_txt,
             font=('Helvetica', 8, 'bold' if i==9 else 'normal'),
             bg='#1e1e2e', fg=label_couleur).grid(row=i+3, column=0, padx=3, pady=2)

    for j in range(4):
        e = tk.Entry(left, width=12, bg=couleur_bg, fg=couleur_fg,
                     insertbackground=couleur_fg, font=('Helvetica', 9))
        e.insert(0, str(default_values[i][j]))
        e.grid(row=i+3, column=j+1, padx=3, pady=2)
        row_entries.append(e)
    entries.append(row_entries)

right = tk.Frame(tab2, bg='#1e1e2e')
right.pack(side='right', fill='both', expand=True, padx=10, pady=10)

title_label = tk.Label(right, text="← Modifiez les mesures et lancez la prévision",
                        font=('Helvetica', 10, 'italic'), bg='#1e1e2e', fg='#aaaaaa')
title_label.pack(pady=3)

fig2, axes2 = plt.subplots(5, 1, figsize=(6, 5))
fig2.patch.set_facecolor('#1e1e2e')
canvas2 = FigureCanvasTkAgg(fig2, master=right)
canvas2.get_tk_widget().pack(fill='both', expand=True)

analyse_frame = tk.Frame(right, bg='#252535')
analyse_frame.pack(fill='x', padx=5, pady=3)
tk.Label(analyse_frame, text="🔍 ANALYSE DES TENDANCES",
         font=('Helvetica', 9, 'bold'), bg='#252535', fg='#89b4fa').pack(anchor='w', padx=10, pady=2)
analyse_label = tk.Label(analyse_frame, text="En attente de prévision...",
                          font=('Courier', 8), bg='#252535', fg='#aaaaaa', justify='left')
analyse_label.pack(anchor='w', padx=10, pady=2)

def lancer_prevision():
    try:
        noms_lignes = [f"{i+1}" for i in range(10)]
        valeurs_brutes, erreurs, avertissements = valider_sequence(entries, noms_lignes)

        if erreurs:
            messagebox.showerror("Valeurs impossibles",
                                 "Les valeurs suivantes sont physiquement impossibles :\n\n" +
                                 "\n".join(erreurs))
            return

        if avertissements:
            msg = "Certaines valeurs sont hors normes OMS :\n\n" + "\n".join(avertissements)
            msg += "\n\nLa prévision sera effectuée mais l'eau présente des anomalies."
            messagebox.showwarning("Attention — Hors normes OMS", msg)

        sequence = []
        for row in valeurs_brutes:
            ph, tds, turb, temp = row
            cond = tds * 2
            sequence.append([ph, tds, turb, cond, temp])

        seq_array = np.array(sequence)
        pred = predict(seq_array)

        now2 = datetime.now()
        title_label.config(
            text=f"Prévision des 24h suivantes — générée à {now2.strftime('%Hh%M')}",
            fg='white')

        for ax in axes2:
            ax.clear()
            ax.set_facecolor('#313244')

        time_labels = [(now2 + timedelta(hours=k)).strftime('%Hh')
                       if k % 4 == 0 else '' for k in range(24)]

        for i, (ax, feat, color, unit) in enumerate(zip(axes2, FEATURES, COLORS, UNITS)):
            y_min = pred[:, i].min() * 0.98
            y_max = pred[:, i].max() * 1.02
            add_oms_zones(ax, feat, y_min, y_max)
            ax.plot(np.arange(24), pred[:, i], color=color, linewidth=2)
            ax.set_ylim(y_min, y_max)
            label = f"{feat} ({unit})" if unit else feat
            ax.set_title(label, color='white', fontsize=8, fontweight='bold')
            ax.set_xticks(range(0, 24, 4))
            ax.set_xticklabels(time_labels[::4], color='white', fontsize=7)
            ax.set_ylabel(unit, color='#aaaaaa', fontsize=7)
            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#555')

        analyse = analyser_tendance(pred)
        analyse_label.config(text=analyse, fg='white')

        fig2.tight_layout(pad=1.0)
        canvas2.draw()

    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite :\n{str(e)}")

tk.Button(left, text="🔮 Lancer la prévision", command=lancer_prevision,
          bg='#89b4fa', fg='#1e1e2e', font=('Helvetica', 11, 'bold'),
          relief='flat', padx=15, pady=8, cursor='hand2').grid(
          row=13, column=0, columnspan=5, pady=10)

# ── Barre sources en bas ──────────────────────────────────────────────────────
source_bar = tk.Frame(root, bg='#111122', pady=4)
source_bar.pack(fill='x', side='bottom')
tk.Label(source_bar,
         text="📚 Sources : OMS — Guidelines for Drinking Water Quality, 4ème édition 2022  |  "
              "USGS — National Water Information System  |  "
              "Aqua-Sens © École Centrale Casablanca — Groupe PLBD 5",
         font=('Helvetica', 7), bg='#111122', fg='#666688').pack()

root.mainloop()