"""
Calculadora de Matriz Jacobiana — Brazo Robótico 2DOF
Newton-Raphson con visualización en tiempo real
Requiere: pip install matplotlib numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════
# CINEMÁTICA DIRECTA E INVERSA (Newton-Raphson + Jacobiana)
# ══════════════════════════════════════════════════════════════════

def cinematica_directa(t1, t2, L1, L2):
    """Posición del extremo dado θ1, θ2."""
    x = L1*np.cos(t1) + L2*np.cos(t1 + t2)
    y = L1*np.sin(t1) + L2*np.sin(t1 + t2)
    return x, y

def jacobiana(t1, t2, L1, L2):
    """Matriz Jacobiana 2x2 del brazo."""
    J = np.array([
        [-L1*np.sin(t1) - L2*np.sin(t1+t2),  -L2*np.sin(t1+t2)],
        [ L1*np.cos(t1) + L2*np.cos(t1+t2),   L2*np.cos(t1+t2)]
    ])
    return J

def newton_raphson_robot(xd, yd, L1, L2, t1_0, t2_0, tol, max_iter):
    """
    Cinemática inversa por Newton-Raphson.
    Devuelve: path de ángulos, historial de residuos, Jacobianas, mensaje.
    """
    # Verificar alcance
    dist = np.sqrt(xd**2 + yd**2)
    if dist > L1 + L2:
        return None, None, None, f"❌ Fuera de alcance: distancia={dist:.2f} > L1+L2={L1+L2:.2f}"
    if dist < abs(L1 - L2) + 1e-6:
        return None, None, None, f"❌ Muy cerca del origen: distancia={dist:.2f} < |L1-L2|={abs(L1-L2):.2f}"

    t = np.array([t1_0, t2_0], dtype=float)
    path      = [t.copy()]
    residuos  = []
    jacobianas = []

    for k in range(max_iter):
        x_cur, y_cur = cinematica_directa(t[0], t[1], L1, L2)
        error = np.array([xd - x_cur, yd - y_cur])
        res   = np.linalg.norm(error)
        residuos.append(res)

        J = jacobiana(t[0], t[1], L1, L2)
        jacobianas.append(J.copy())

        if res < tol:
            return path, residuos, jacobianas, f"✅ Convergió en {k} iteraciones  |  residuo = {res:.2e}"

        det = np.linalg.det(J)
        if abs(det) < 1e-10:
            return path, residuos, jacobianas, f"⚠️ Jacobiana singular en iteración {k}  |  det={det:.2e}"

        delta = np.linalg.solve(J, error)
        t = t + delta
        path.append(t.copy())

    x_cur, y_cur = cinematica_directa(t[0], t[1], L1, L2)
    res = np.linalg.norm([xd - x_cur, yd - y_cur])
    residuos.append(res)
    return path, residuos, jacobianas, f"❌ No convergió en {max_iter} iteraciones  |  residuo final = {res:.2e}"


# ══════════════════════════════════════════════════════════════════
# INTERFAZ GRÁFICA
# ══════════════════════════════════════════════════════════════════

class RobotCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora Jacobiana — Brazo Robótico 2DOF")
        self.root.configure(bg="#1e1e2e")
        self.root.geometry("1280x780")
        self.root.resizable(True, True)

        self._anim_data = None
        self._ani       = None
        self._build_ui()

    # ── Layout principal ─────────────────────────────────────────
    def _build_ui(self):
        # Panel izquierdo (controles)
        left = tk.Frame(self.root, bg="#1e1e2e", width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)
        left.pack_propagate(False)

        # Panel derecho (gráficas)
        right = tk.Frame(self.root, bg="#1e1e2e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,12), pady=12)

        self._build_controls(left)
        self._build_plots(right)

    # ── Controles ────────────────────────────────────────────────
    def _build_controls(self, parent):
        style_label = {"bg":"#1e1e2e", "fg":"#cdd6f4", "font":("Consolas",10)}
        style_title = {"bg":"#1e1e2e", "fg":"#89b4fa", "font":("Consolas",11,"bold")}
        style_entry = {"bg":"#313244", "fg":"#cdd6f4", "insertbackground":"#cdd6f4",
                       "relief":"flat", "font":("Consolas",11), "width":12}

        def section(text):
            tk.Label(parent, text=text, **style_title).pack(anchor="w", pady=(14,2))
            tk.Frame(parent, bg="#45475a", height=1).pack(fill=tk.X, pady=(0,6))

        def row(label, default):
            f = tk.Frame(parent, bg="#1e1e2e")
            f.pack(fill=tk.X, pady=3)
            tk.Label(f, text=label, width=22, anchor="w", **style_label).pack(side=tk.LEFT)
            e = tk.Entry(f, **style_entry)
            e.insert(0, default)
            e.pack(side=tk.LEFT)
            return e

        # ── Eslabones
        section("⚙  Eslabones")
        self.e_L1 = row("L1  (longitud 1):", "100")
        self.e_L2 = row("L2  (longitud 2):", "80")

        # ── Posición objetivo
        section("🎯  Posición objetivo")
        self.e_xd = row("x  destino:", "120")
        self.e_yd = row("y  destino:", "80")

        # ── Ángulos iniciales
        section("📐  Ángulos iniciales (°)")
        self.e_t1 = row("θ₁  inicial:", "30")
        self.e_t2 = row("θ₂  inicial:", "45")

        # ── Newton-Raphson
        section("🔢  Newton-Raphson")
        self.e_tol     = row("Tolerancia:", "1e-6")
        self.e_maxiter = row("Máx. iteraciones:", "50")

        # ── Botón calcular
        tk.Frame(parent, bg="#1e1e2e").pack(pady=10)
        btn = tk.Button(parent, text="▶  CALCULAR",
                        bg="#89b4fa", fg="#1e1e2e",
                        font=("Consolas",12,"bold"),
                        relief="flat", cursor="hand2",
                        activebackground="#b4befe",
                        command=self.calcular)
        btn.pack(fill=tk.X, ipady=8)

        self.btn_anim = tk.Button(parent, text="▷  Animar movimiento",
                              bg="#a6e3a1", fg="#1e1e2e",
                              font=("Consolas",10,"bold"),
                              relief="flat", cursor="hand2",
                              activebackground="#cba6f7",
                              state=tk.DISABLED,
                              command=self.animar)
        self.btn_anim.pack(fill=tk.X, ipady=6, pady=(6,0))

        btn_clear = tk.Button(parent, text="↺  Limpiar",
                              bg="#313244", fg="#cdd6f4",
                              font=("Consolas",10),
                              relief="flat", cursor="hand2",
                              command=self.limpiar)
        btn_clear.pack(fill=tk.X, ipady=4, pady=(4,0))

        # ── Resultado / estado
        tk.Frame(parent, bg="#1e1e2e").pack(pady=8)
        self.lbl_status = tk.Label(parent, text="Ingresa parámetros y presiona Calcular",
                                   bg="#1e1e2e", fg="#a6e3a1",
                                   font=("Consolas",9), wraplength=290, justify="left")
        self.lbl_status.pack(anchor="w")

        # ── Tabla de iteraciones
        section("📋  Iteraciones")
        cols = ("iter","θ₁(°)","θ₂(°)","x","y","residuo")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)
        for c, w in zip(cols, [35,60,60,65,65,80]):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background="#313244", foreground="#cdd6f4",
                        fieldbackground="#313244", font=("Consolas",8),
                        rowheight=18)
        style.configure("Treeview.Heading",
                        background="#45475a", foreground="#89b4fa",
                        font=("Consolas",8,"bold"))

        sb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)

    # ── Gráficas ─────────────────────────────────────────────────
    def _build_plots(self, parent):
        self.fig = plt.figure(figsize=(9, 6.5), facecolor="#1e1e2e")
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.38, wspace=0.32,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.08)

        self.ax_robot  = self.fig.add_subplot(gs[0, 0])   # brazo en posición final
        self.ax_tray   = self.fig.add_subplot(gs[0, 1])   # trayectoria del extremo
        self.ax_conv   = self.fig.add_subplot(gs[1, 0])   # convergencia del residuo
        self.ax_jac    = self.fig.add_subplot(gs[1, 1])   # mapa de calor Jacobiana

        for ax in (self.ax_robot, self.ax_tray, self.ax_conv, self.ax_jac):
            ax.set_facecolor("#181825")
            for sp in ax.spines.values(): sp.set_color("#45475a")
            ax.tick_params(colors="#6c7086", labelsize=8)
            ax.xaxis.label.set_color("#cdd6f4")
            ax.yaxis.label.set_color("#cdd6f4")
            ax.title.set_color("#89b4fa")

        self._placeholder_plots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def _placeholder_plots(self):
        for ax, t in [(self.ax_robot, "Brazo robótico"),
                      (self.ax_tray,  "Trayectoria del extremo"),
                      (self.ax_conv,  "Convergencia del residuo"),
                      (self.ax_jac,   "Jacobiana (última iteración)")]:
            ax.clear()
            ax.set_facecolor("#181825")
            ax.set_title(t, fontsize=9, color="#89b4fa", pad=5)
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    color="#45475a", fontsize=18, transform=ax.transAxes)

    # ── Leer entradas ────────────────────────────────────────────
    def _get_float(self, entry, name):
        try:
            return float(entry.get())
        except ValueError:
            raise ValueError(f"Valor inválido en '{name}': '{entry.get()}'")

    # ── Calcular ─────────────────────────────────────────────────
    def calcular(self):
        try:
            L1       = self._get_float(self.e_L1, "L1")
            L2       = self._get_float(self.e_L2, "L2")
            xd       = self._get_float(self.e_xd, "x destino")
            yd       = self._get_float(self.e_yd, "y destino")
            t1_0     = np.radians(self._get_float(self.e_t1, "θ₁ inicial"))
            t2_0     = np.radians(self._get_float(self.e_t2, "θ₂ inicial"))
            tol      = self._get_float(self.e_tol, "Tolerancia")
            max_iter = int(self._get_float(self.e_maxiter, "Máx. iteraciones"))
        except ValueError as e:
            messagebox.showerror("Error de entrada", str(e))
            return

        path, residuos, jacobianas, msg = newton_raphson_robot(
            xd, yd, L1, L2, t1_0, t2_0, tol, max_iter
        )

        color_msg = "#a6e3a1" if "✅" in msg else "#f38ba8" if "❌" in msg else "#fab387"
        self.lbl_status.config(text=msg, fg=color_msg)

        if path is None:
            return

        # Guardar para animación
        self._anim_data = (path, residuos, jacobianas, xd, yd, L1, L2)
        self._ani       = None
        self.btn_anim.config(state=tk.NORMAL)

        self._fill_table(path, residuos, L1, L2)
        self._draw_robot(path, residuos, jacobianas, xd, yd, L1, L2)

    # ── Tabla de iteraciones ─────────────────────────────────────
    def _fill_table(self, path, residuos, L1, L2):
        for row in self.tree.get_children():
            self.tree.delete(row)

        for i, (t1, t2) in enumerate(path):
            x, y = cinematica_directa(t1, t2, L1, L2)
            res  = residuos[i] if i < len(residuos) else 0.0
            tag  = "final" if i == len(path)-1 else ("even" if i%2==0 else "odd")
            self.tree.insert("", "end", values=(
                i,
                f"{np.degrees(t1):.3f}",
                f"{np.degrees(t2):.3f}",
                f"{x:.3f}",
                f"{y:.3f}",
                f"{res:.2e}"
            ), tags=(tag,))

        self.tree.tag_configure("even",  background="#313244")
        self.tree.tag_configure("odd",   background="#2a2a3e")
        self.tree.tag_configure("final", background="#1c3a1c", foreground="#a6e3a1")

    # ── Gráficas de resultados ───────────────────────────────────
    def _draw_robot(self, path, residuos, jacobianas, xd, yd, L1, L2):
        t_final = path[-1]
        t1f, t2f = t_final

        lim = (L1 + L2) * 1.15

        # ── 1. Brazo en posición final ────────────────────────────
        ax = self.ax_robot
        ax.clear(); ax.set_facecolor("#181825")
        ax.set_title("Brazo robótico — posición final", fontsize=9, color="#89b4fa", pad=5)

        # Workspace (espacio de trabajo)
        theta_ws = np.linspace(0, 2*np.pi, 200)
        ax.fill(( L1+L2)*np.cos(theta_ws), ( L1+L2)*np.sin(theta_ws),
                color="#89b4fa", alpha=0.04)
        ax.fill((abs(L1-L2))*np.cos(theta_ws), (abs(L1-L2))*np.sin(theta_ws),
                color="#1e1e2e", alpha=1.0)
        ax.plot(( L1+L2)*np.cos(theta_ws), ( L1+L2)*np.sin(theta_ws),
                color="#45475a", lw=0.8, linestyle="--")

        # Eslabones
        O  = np.array([0.0, 0.0])
        A  = L1 * np.array([np.cos(t1f), np.sin(t1f)])
        E  = A + L2 * np.array([np.cos(t1f+t2f), np.sin(t1f+t2f)])

        ax.plot([O[0],A[0]], [O[1],A[1]], color="#89b4fa", lw=4,
                solid_capstyle="round", label=f"L1={L1}")
        ax.plot([A[0],E[0]], [A[1],E[1]], color="#cba6f7", lw=4,
                solid_capstyle="round", label=f"L2={L2}")

        # Nodos
        for pt, col, sz in [(O,"#f38ba8",80),(A,"#fab387",60),(E,"#a6e3a1",80)]:
            ax.scatter(*pt, color=col, s=sz, zorder=6)

        # Objetivo
        ax.scatter(xd, yd, color="#f9e2af", s=120, marker="*",
                   zorder=7, label=f"Objetivo ({xd},{yd})")
        ax.plot([E[0],xd],[E[1],yd], color="#f9e2af", lw=1,
                linestyle=":", alpha=0.6)

        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.1, color="#45475a")
        ax.axhline(0, color="#45475a", lw=0.5)
        ax.axvline(0, color="#45475a", lw=0.5)
        ax.legend(fontsize=7, loc="upper right",
                  labelcolor="#cdd6f4", facecolor="#313244", edgecolor="#45475a")
        for sp in ax.spines.values(): sp.set_color("#45475a")
        ax.tick_params(colors="#6c7086", labelsize=7)

        # ── 2. Trayectoria del extremo ────────────────────────────
        ax = self.ax_tray
        ax.clear(); ax.set_facecolor("#181825")
        ax.set_title("Trayectoria del extremo (E)", fontsize=9, color="#89b4fa", pad=5)

        xs = [cinematica_directa(t[0], t[1], L1, L2)[0] for t in path]
        ys = [cinematica_directa(t[0], t[1], L1, L2)[1] for t in path]

        # Gradiente de color por iteración
        n = len(xs)
        for i in range(n-1):
            c = plt.cm.plasma(i / max(n-1, 1))
            ax.plot(xs[i:i+2], ys[i:i+2], color=c, lw=2)

        ax.scatter(xs[0],  ys[0],  color="#fab387", s=80, zorder=6, label="Inicio")
        ax.scatter(xs[-1], ys[-1], color="#a6e3a1", s=80, zorder=6, label="Final")
        ax.scatter(xd, yd, color="#f9e2af", s=120, marker="*",
                   zorder=7, label="Objetivo")

        # Números de iteración
        for i,(x,y) in enumerate(zip(xs,ys)):
            ax.annotate(str(i),(x,y), fontsize=7, color="#6c7086",
                        xytext=(4,4), textcoords="offset points")

        ax.grid(True, alpha=0.1, color="#45475a")
        ax.legend(fontsize=7, labelcolor="#cdd6f4",
                  facecolor="#313244", edgecolor="#45475a")
        for sp in ax.spines.values(): sp.set_color("#45475a")
        ax.tick_params(colors="#6c7086", labelsize=7)

        # ── 3. Convergencia del residuo ───────────────────────────
        ax = self.ax_conv
        ax.clear(); ax.set_facecolor("#181825")
        ax.set_title("Convergencia del residuo ||error||", fontsize=9, color="#89b4fa", pad=5)

        iters = np.arange(len(residuos))
        ax.semilogy(iters, residuos, color="#89b4fa", lw=2, marker="o",
                    ms=5, markerfacecolor="#cba6f7")
        ax.fill_between(iters, residuos, alpha=0.1, color="#89b4fa")
        ax.axhline(y=residuos[-1], color="#a6e3a1", lw=1,
                   linestyle="--", alpha=0.7, label=f"Final: {residuos[-1]:.2e}")

        ax.set_xlabel("Iteración", fontsize=8)
        ax.set_ylabel("Residuo (log)", fontsize=8)
        ax.grid(True, alpha=0.15, color="#45475a")
        ax.legend(fontsize=7, labelcolor="#cdd6f4",
                  facecolor="#313244", edgecolor="#45475a")
        for sp in ax.spines.values(): sp.set_color("#45475a")
        ax.tick_params(colors="#6c7086", labelsize=7)
        ax.yaxis.label.set_color("#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")

        # ── 4. Mapa de calor — Jacobiana final ────────────────────
        ax = self.ax_jac
        ax.clear(); ax.set_facecolor("#181825")
        ax.set_title("Jacobiana — última iteración", fontsize=9, color="#89b4fa", pad=5)

        J_last = jacobianas[-1]
        im = ax.imshow(J_last, cmap="coolwarm", aspect="auto")
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        labels_r = ["∂x/∂θ₁", "∂y/∂θ₁"]
        labels_c = ["∂x/∂θ₂", "∂y/∂θ₂"]
        ax.set_xticks([0,1]); ax.set_xticklabels(["col θ₁","col θ₂"],
                                                   color="#cdd6f4", fontsize=8)
        ax.set_yticks([0,1]); ax.set_yticklabels(["fila x","fila y"],
                                                   color="#cdd6f4", fontsize=8)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{J_last[i,j]:.3f}",
                        ha="center", va="center",
                        color="#1e1e2e", fontsize=11, fontweight="bold")

        det = np.linalg.det(J_last)
        ax.set_xlabel(f"det(J) = {det:.4f}", fontsize=8, color="#cdd6f4")
        for sp in ax.spines.values(): sp.set_color("#45475a")

        self.canvas.draw()

    # ── Animación ────────────────────────────────────────────────
    def animar(self):
        if self._anim_data is None:
            return

        path, residuos, jacobianas, xd, yd, L1, L2 = self._anim_data
        lim = (L1 + L2) * 1.15

        # Detener animación previa si existe
        if self._ani is not None:
            self._ani.event_source.stop()
            self._ani = None

        ax = self.ax_robot
        ax.clear()
        ax.set_facecolor("#181825")
        ax.set_title("Brazo robótico — animación", fontsize=9, color="#89b4fa", pad=5)

        # Espacio de trabajo
        theta_ws = np.linspace(0, 2*np.pi, 200)
        ax.fill((L1+L2)*np.cos(theta_ws), (L1+L2)*np.sin(theta_ws),
                color="#89b4fa", alpha=0.04)
        ax.fill(abs(L1-L2)*np.cos(theta_ws), abs(L1-L2)*np.sin(theta_ws),
                color="#1e1e2e", alpha=1.0)
        ax.plot((L1+L2)*np.cos(theta_ws), (L1+L2)*np.sin(theta_ws),
                color="#45475a", lw=0.8, linestyle="--")

        # Objetivo fijo
        ax.scatter(xd, yd, color="#f9e2af", s=140, marker="*", zorder=7,
                   label=f"Objetivo ({xd},{yd})")

        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.1, color="#45475a")
        ax.axhline(0, color="#45475a", lw=0.5)
        ax.axvline(0, color="#45475a", lw=0.5)
        for sp in ax.spines.values(): sp.set_color("#45475a")
        ax.tick_params(colors="#6c7086", labelsize=7)

        # Traza del extremo
        tray_line, = ax.plot([], [], color="#cba6f7", lw=1,
                             linestyle="--", alpha=0.5, zorder=3)

        # Eslabones animados
        line_OA, = ax.plot([], [], color="#89b4fa", lw=5,
                           solid_capstyle="round", zorder=4)
        line_AE, = ax.plot([], [], color="#cba6f7", lw=5,
                           solid_capstyle="round", zorder=4)

        # Nodos animados
        dot_O = ax.scatter([], [], color="#f38ba8", s=90,  zorder=6)
        dot_A = ax.scatter([], [], color="#fab387", s=70,  zorder=6)
        dot_E = ax.scatter([], [], color="#a6e3a1", s=90,  zorder=6)

        # Texto de iteración
        txt_iter = ax.text(0.03, 0.96, "", transform=ax.transAxes,
                           fontsize=9, color="#cdd6f4",
                           verticalalignment="top",
                           fontfamily="Consolas")

        # Línea de error
        err_line, = ax.plot([], [], color="#f9e2af", lw=1,
                            linestyle=":", alpha=0.7, zorder=3)

        tray_x, tray_y = [], []

        def init_anim():
            line_OA.set_data([], [])
            line_AE.set_data([], [])
            tray_line.set_data([], [])
            err_line.set_data([], [])
            dot_O.set_offsets(np.empty((0, 2)))
            dot_A.set_offsets(np.empty((0, 2)))
            dot_E.set_offsets(np.empty((0, 2)))
            txt_iter.set_text("")
            tray_x.clear(); tray_y.clear()
            return line_OA, line_AE, tray_line, err_line, dot_O, dot_A, dot_E, txt_iter

        def update_anim(frame):
            # Pausa extra al final para ver la posición final
            idx  = min(frame, len(path) - 1)
            t1, t2 = path[idx]

            O = np.array([0.0, 0.0])
            A = L1 * np.array([np.cos(t1), np.sin(t1)])
            E = A + L2 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])

            line_OA.set_data([O[0], A[0]], [O[1], A[1]])
            line_AE.set_data([A[0], E[0]], [A[1], E[1]])
            err_line.set_data([E[0], xd],  [E[1], yd])

            tray_x.append(E[0]); tray_y.append(E[1])
            tray_line.set_data(tray_x, tray_y)

            dot_O.set_offsets([O])
            dot_A.set_offsets([A])
            dot_E.set_offsets([E])

            res = residuos[idx] if idx < len(residuos) else residuos[-1]
            es_final = (idx == len(path) - 1)
            estado   = " ✅" if es_final else ""
            txt_iter.set_text(
                f"iter {idx}{estado}\n"
                f"θ₁={np.degrees(t1):>7.3f}°\n"
                f"θ₂={np.degrees(t2):>7.3f}°\n"
                f"err={res:.2e}"
            )
            # Resaltar extremo en verde al llegar
            dot_E.set_color("#a6e3a1" if not es_final else "#00ff88")

            return line_OA, line_AE, tray_line, err_line, dot_O, dot_A, dot_E, txt_iter

        total_frames = len(path) + 6   # 6 frames extra de pausa al final

        import matplotlib.animation as animation
        self._ani = animation.FuncAnimation(
            self.fig, update_anim,
            frames=total_frames,
            init_func=init_anim,
            interval=350,
            blit=False,
            repeat=True
        )

        self.btn_anim.config(text="⏹  Detener", command=self._stop_anim,
                             bg="#f38ba8")
        self.canvas.draw()

    def _stop_anim(self):
        if self._ani is not None:
            self._ani.event_source.stop()
            self._ani = None
        self.btn_anim.config(text="▷  Animar movimiento", command=self.animar,
                             bg="#a6e3a1")
        # Restaurar gráfica estática
        if self._anim_data:
            path, residuos, jacobianas, xd, yd, L1, L2 = self._anim_data
            self._draw_robot(path, residuos, jacobianas, xd, yd, L1, L2)

    # ── Limpiar ──────────────────────────────────────────────────
    def limpiar(self):
        if self._ani is not None:
            self._ani.event_source.stop()
            self._ani = None
        self._anim_data = None
        self.btn_anim.config(state=tk.DISABLED, text="▷  Animar movimiento",
                             command=self.animar, bg="#a6e3a1")
        self._placeholder_plots()
        self.canvas.draw()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.lbl_status.config(
            text="Ingresa parámetros y presiona Calcular", fg="#a6e3a1"
        )


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = RobotCalculator(root)
    root.mainloop()