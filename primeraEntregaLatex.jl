using Distributions
using StatsPlots
using Random
using Statistics
using Plots
using LinearAlgebra


h=1
m = 0.5 #para que 2m=1

#construimos nuestra barrera
#literal las dimensiones del rectangulo
a= 5.0
V0 = 5.0

#rangos de energía que manejaremos
E_min = 0.1
E_max = 20.0 #sobre V0
dE = 0.01

#definiremos el rango donde se graficará la función de onda
x_min = -10.0
x_max = 12.0
dx = 0.01 

#estado energético a mostrar por corrida
#probaremos con valroes mayores y menores a V0
estadoE = 5.0

##----------------NUMERO DE ONDA K / Q / κ ------
function kE(E)
    return sqrt(Complex(2*m* max(E,0.0)))/h
end

function qE(E, V0)
    ΔE = E - V0
    if ΔE >= 0
        # E > V0  → q real (oscilatorio)
        return sqrt(Complex(2 * m * ΔE)) / h
    else
        # E < V0  → q imaginario (túnel)
        κ = sqrt(Complex(2 * m * (V0 - E))) / h
        return 1im * κ
    end
end


#para el caso imaginario, la función de q ya regresa lo que valdría kappa

function rYt(E, V0, a)
    k = kE(E)
    q = qE(E, V0)
    qa = q * a

    Den = 2 * k * q * cos(qa) - 1im * (k^2 + q^2) * sin(qa)
    r = 1im * (k^2 - q^2) * sin(qa) / Den
    t = (2 * k * q * exp(-1im * k * a)) / Den

    return r, t
end


#-----------------------COEFICIENTES--------
function rt_numerico(E, V0, a, m, h)
    k = kE(E)
    q = qE(E, V0)

    # Evita singularidad cuando q ≈ 0 (E ≈ V0)
    if abs(q) < 1e-10
        return 0.0 + 0.0im, 1.0 + 0.0im, (0.0 + 0.0im, 0.0 + 0.0im)
    end

    if isreal(q)  # E > V0
        q = real(q)
        M = zeros(ComplexF64, 4, 4)
        b = zeros(ComplexF64, 4)

        eipa = exp(1im * q * a)
        eima = exp(-1im * q * a)
        eika = exp(1im * k * a)

        # Condiciones de continuidad
        M[1, :] = [1, -1, -1, 0]
        b[1] = -1

        M[2, :] = [-1im * k, -1im * q, 1im * q, 0]
        b[2] = -1im * k

        M[3, :] = [0, eipa, eima, -eika]
        b[3] = 0

        M[4, :] = [0, 1im * q * eipa, -1im * q * eima, -1im * k * eika]
        b[4] = 0

        sol = M \ b
        r, A, B, t = sol
        inner = (A, B)

    else  # E < V0 (túnel)
        κ = imag(q)
        M = zeros(ComplexF64, 4, 4)
        b = zeros(ComplexF64, 4)

        epa  = exp(κ * a)
        ema  = exp(-κ * a)
        eika = exp(1im * k * a)

        M[1, :] = [1, -1, -1, 0]
        b[1] = -1

        M[2, :] = [-1im * k, -κ, κ, 0]
        b[2] = -1im * k

        M[3, :] = [0, epa, ema, -eika]
        b[3] = 0

        M[4, :] = [0, κ * epa, -κ * ema, -1im * k * eika]
        b[4] = 0

        sol = M \ b
        r, C, D, t = sol
        inner = (C, D)
    end

    return r, t, inner
end



function corrienteJ(ampli)
    return ((h* k) / m) * (abs(ampli)^2)
end


function rt_a_RT( r, t)
    R = (abs.(r)).^2
    T = (abs.(t)).^2
    return R, T
end

#-----------------CALCULAMOS LA PSI E NCADA REGION
function psiXregiones(x, E, V0, a, rt_internas)
    
    k = kE(E)
    q = qE(E, V0)
    r, t , internas = rt_internas #coeficientes (A, B) o (C, D)

    psi = zeros(ComplexF64, length(x))

    #definimos las regiones
    antes = (x .< 0.0)
    dentro = (x .>= 0.0) .& (x .<= a) #aquí implementamos continuidad
    despues = (x .> a)

    #región I ..... x<a
    psi[antes] = exp.(1im * k .* x[antes]) .+ r .* exp.(-1im * k .* x[antes]) 

    #región II ..... 0 ≤ x ≤ a 

    if isreal(q)
        A, B  = internas
        psi[dentro] = A .* exp.(1im * q .* x[dentro]) .+ B .*exp.(-1im * q .* x[dentro] ) 

    else
        C, D = internas 
        κ = imag(q)
        psi[dentro]= C .* exp.(κ .*x[dentro]) .+ D .* exp.(-κ .* x[dentro] )
        
    end

    #región III .... x>a 
    psi[despues]= t .* exp.(1im * k .* x[despues])

    return psi
end



#vectres de energía y otras cosas para la animación
energias = range(E_min, E_max; step=0.01)

r_an = zeros(ComplexF64, length(energias))
t_an = zeros(ComplexF64, length(energias))

r_num = zeros(ComplexF64, length(energias))
t_num = zeros(ComplexF64, length(energias))

for (i, E) in enumerate(energias)
    r_an[i], t_an[i] = rYt(E, V0, a)
    r_num[i], t_num[i] = rt_numerico(E, V0, a, m, h)
end

R_an, T_an = rt_a_RT(r_an, t_an)
R_num, T_num = rt_a_RT(r_num, t_num)


#llamamos a las funciones de coefs para calcularlos con la energia a evaluar
for (i, E) in enumerate(energias)
    if isapprox(E, V0; atol=1e-10)
        r_an[i] = 0; t_an[i] = 1
        r_num[i] = 0; t_num[i] = 1
        continue
    end
    r_an[i], t_an[i] = rYt(E, V0, a)
    r_num[i], t_num[i], _ = rt_numerico(E, V0, a, m, h)
end

R_an, T_an = rt_a_RT(r_an, t_an)
R_num, T_num = rt_a_RT(r_num, t_num)



#-------plots coefs

p1 = plot(
    energias, R_num,
    label = "R_num",
    lw = 2,
    xlabel = "Energía",
    ylabel = "Coeficientes",
    title = "Barrera de potencial con V₀=$(V0) y a=$(a)",
    color=:aquamarine4,
    legend = true,
    grid = true,
)

plot!(energias, T_num, label = "T_num", lw = 2, color=:darkmagenta)
plot!(energias, (T_an+R_an), label= "R + T", lw= 2, color=:darkorange3, style= :dash)
vline!([V0], label = "V₀", color = :black, lw = 10, alpha = 0.3, cmap=:maroon)


# ---------- le hacemos zoom porq no se ve

p2 = plot(
    energias, R_an,
    xlims = (4,7.5),
    label = "R ",
    lw = 2,
    xlabel = "Energía",
    ylabel = "Coeficientes",
    title = "Barrera de potencial para x ϵ (3, 10)",
    color=:aquamarine4,
    legend = true,
    grid = true,
)

plot!(energias, T_an, label = "T ", lw = 2, color=:darkmagenta)
vline!([V0], label = "V₀", lw = 10, alpha = 0.3, color=:maroon)

#savefig(p1, "coefs.png")
#savefig(p2, "coefsZoom.png")

#partes real e imaginarias


# --- Escoge la energía a visualizar ---
E_plot = 2.0   # Puedes probar con E < V0, E ≈ V0 o E > V0

# Calcula la función de onda ψ(x)
r_num, t_num, internas = rt_numerico(E_plot, V0, a, m, h)
x_vals = range(x_min, x_max; step=dx)
psi = psiXregiones(x_vals, E_plot, V0, a, (r_num, t_num, internas))

# --- Creamos tres subplots ---
p6 = plot(x_vals, real.(psi), lw=2, color=:darkgreen,
    xlabel="x", ylabel="ψ(x)", title="Partes real e imaginaria de ψ(x)",
    grid=true, legend=false)
vline!([0, a], color=:hotpink1, lw=2, alpha=0.15)
plot!([0, a], [maximum(real.(psi)), maximum(real.(psi))],
      fillrange=minimum(real.(psi)), fillalpha=0.15,
      fillcolor=:hotpink1, linealpha=0)
plot!(x_vals, imag.(psi), lw=2, color=:deeppink2)

"""
plot!(x_vals, imag.(psi), lw=2, color=:deeppink2,
    xlabel="x", ylabel="Im(ψ)", title="Parte imaginaria de ψ(x)",
    grid=true, legend=false)
    
vline!([0, a], color=:hotpink1, lw=2, alpha=0.15)
plot!([0, a], [maximum(imag.(psi)), maximum(imag.(psi))],
      fillrange=minimum(imag.(psi)), fillalpha=0.15,
      fillcolor=:hotpink1, linealpha=0)
"""
p8 = plot(x_vals, abs.(psi), lw=2, color=:darkmagenta,
    xlabel="x", ylabel="|ψ(x)|", title="Módulo de ψ(x)",
    grid=true, legend=false)
vline!([0, a], color=:hotpink1, lw=2, alpha=0.15)
plot!([0, a], [maximum(abs.(psi)), maximum(abs.(psi))],
      fillrange=0, fillalpha=0.15, fillcolor=:hotpink1, linealpha=0)

# --- Muestra las tres gráficas juntas ---
plot68= plot(p6, p8, layout=(2,1), size=(800,600))

#savefig(plot68, "ImReAbs.png")

using Plots

# Energías a analizar (puedes cambiar o agregar)
E_vals = [2.0, 5.01, 7.0, 15.0]   # incluye E < V0, ≈V0 y >V0
x_vals = range(x_min, x_max; step=dx)

frames = []  # guardará las gráficas

for E_plot in E_vals
    # --- Calcula ψ(x) ---
    r_num, t_num, internas = rt_numerico(E_plot, V0, a, m, h)
    psi = psiXregiones(x_vals, E_plot, V0, a, (r_num, t_num, internas))

    # --- Gráfica combinada de Re(ψ), Im(ψ) y |ψ| ---
    p10 = plot(
        x_vals, real.(psi),
        lw = 2, color = :darkgreen,
        xlabel = "x", ylabel = "ψ(x)",
        title = "ψ(x) para E = $(round(E_plot, digits=2))",
        grid = true, legend = :topright,
        size = (800, 500)
    )

    plot!(x_vals, imag.(psi), lw=2, color=:deeppink2, label="Im(ψ)")
    plot!(x_vals, abs.(psi), lw=2, color=:darkmagenta, label="|ψ|")

    # --- Fondo de la barrera ---
    vline!([0, a], color=:hotpink1, lw=2, alpha=0.3, label="")
# Relleno uniforme de la barrera entre 0 y a
x_fill = [0, a]
y_fill = [maximum(abs.(psi)) * 1.1, maximum(abs.(psi)) * 1.1]  # línea plana
plot!(x_fill, y_fill,
      fillrange = -maximum(abs.(psi)) * 1.1,  # rellena hasta el valor negativo
      fillcolor = :hotpink1, fillalpha = 0.2, linealpha = 0, label = "")


    push!(frames, p10)
end

# --- Muestra las cuatro gráficas en una cuadrícula 2x2 ---
plot_all = plot(frames..., layout=(2,2), size=(900,700))
display(plot_all)


#savefig(plot_all, "barrido.png")

anim = @animate for E_plot in range(1.0, 10.0; length=60)
    r_num, t_num, internas = rt_numerico(E_plot, V0, a, m, h)
    psi = psiXregiones(x_vals, E_plot, V0, a, (r_num, t_num, internas))

    plot(x_vals, abs.(psi),
        lw = 2, color = :purple,
        xlabel = "x", ylabel = "|ψ(x)|",
        title = "Evolución de |ψ(x)|   (E = $(round(E_plot, digits=2)))",
        ylim = (0, 2.2),
        legend = false, grid = true, size = (800, 500)
    )
    plot!([0, a], [2.2, 2.2], fillrange = 0, fillalpha = 0.2,
          fillcolor = :hotpink1, linealpha = 0)
    vline!([0, a], color=:hotpink1, lw=2, alpha=0.4)
end

# Guardar el GIF (opcional)
#gif(anim, "psi_barrera.mp4", fps=10)


