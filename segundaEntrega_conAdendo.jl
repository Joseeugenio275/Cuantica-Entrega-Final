using LinearAlgebra
using Plots




ħ = 1.0
m = 0.5        # => 2m = 1
V0 = 5.0
a  = 9.0       # ancho de la barrera

energy_from_k(k, m, ħ) = ħ^2 * k^2 / (2m) 

function scattering_coeffs(k; V0=V0, a=a, m=m, ħ=ħ)
    E  = energy_from_k(k, m, ħ)
    ΔE = E - V0

    M = zeros(ComplexF64, 4, 4)
    b = zeros(ComplexF64, 4)

    if ΔE >= 0
        # ----- E > V0 -----
        q = sqrt(2m * ΔE) / ħ

        eika = exp(1im * k * a)
        eipa = exp(1im * q * a)
        eima = exp(-1im * q * a)

        # x = 0
        M[1, :] = [ 1, -1, -1,  0 ]                    # 1 + r = A + B
        b[1]    = -1
        M[2, :] = [ -1im*k, -1im*q,  1im*q, 0 ]         # ik(1-r) = iq(A-B)
        b[2]    = -1im * k

        # x = a
        M[3, :] = [ 0, eipa, eima, -eika ]             # A e^{iq a} + B e^{-iq a} = t e^{ik a}
        b[3]    = 0
        M[4, :] = [ 0, 1im*q*eipa, -1im*q*eima, -1im*k*eika ]  # iq(Ae^{iq a}-Be^{-iq a}) = ik t e^{ik a}
        b[4]    = 0

        u = M \ b
        r, A, B, t = u
        return E, :above, (r, A, B, t)

    else
        # ----- E < V0 -----
        κ = sqrt(2m * (V0 - E)) / ħ

        eika = exp(1im * k * a)
        epa  = exp(κ * a)
        ema  = exp(-κ * a)

        # x = 0
        M[1, :] = [ 1, -1, -1,  0 ]                    # 1 + r = C + D
        b[1]    = -1
        M[2, :] = [ -1im*k, -κ,  κ,  0 ]               # ik(1-r) = κ(C-D)
        b[2]    = -1im * k

        # x = a
        M[3, :] = [ 0, epa, ema, -eika ]
        b[3]    = 0
        M[4, :] = [ 0, κ*epa, -κ*ema, -1im*k*eika ]
        b[4]    = 0

        u = M \ b
        r, C, D, t = u
        return E, :below, (r, C, D, t)
    end
end

function psi_stationary(x::AbstractVector, k; V0=V0, a=a, m=m, ħ=ħ)
    E, regime, coeffs = scattering_coeffs(k; V0=V0, a=a, m=m, ħ=ħ)
    ψ = zeros(ComplexF64, length(x))

    antes   = x .< 0.0
    dentro  = (x .>= 0.0) .& (x .<= a)
    despues = x .> a         

    if regime == :above
        r, A, B, t = coeffs
        q = sqrt(2m * (E - V0)) / ħ

        ψ[antes]   .= exp.(1im * k .* x[antes]) .+ r .* exp.(-1im * k .* x[antes])
        ψ[dentro]  .= A .* exp.(1im * q .* x[dentro]) .+ B .* exp.(-1im * q .* x[dentro])
        ψ[despues] .= t .* exp.(1im * k .* x[despues])

    else
        r, C, D, t = coeffs
        κ = sqrt(2m * (V0 - E)) / ħ

        ψ[antes]   .= exp.(1im * k .* x[antes]) .+ r .* exp.(-1im * k .* x[antes])
        ψ[dentro]  .= C .* exp.(κ .* x[dentro]) .+ D .* exp.(-κ .* x[dentro])
        ψ[despues] .= t .* exp.(1im * k .* x[despues])
    end

    return ψ, E
end


# Gaussiana en k
function phi_k(k; k0, σk, x0)
    norm = (1 / (2π * σk^2))^(1/4)
    # factor e^{-ik x0} desplaza el paquete a x0 en t=0
    return norm * exp(-(k - k0)^2 / (4σk^2)) * exp(-1im * k * x0)
end



function psi_paquete(x::AbstractVector, t;
                        k0,
                        σk,
                        x0,
                        V0=V0, a=a, m=m, ħ=ħ,
                        Nk=400)

    kmin = max(0.0, k0 - 5σk)
    kmax = k0 + 5σk
    ks   = range(kmin, kmax, length=Nk)
    Δk   = step(ks)

    ψ = zeros(ComplexF64, length(x))
    pref = sqrt(Δk / (2π))   # factor tipo FT

    for k in ks
        φ   = phi_k(k; k0=k0, σk=σk, x0=x0)
        ψk, E = psi_stationary(x, k; V0=V0, a=a, m=m, ħ=ħ)
        phase = exp(-1im * E * t / ħ)
        ψ .+= pref * φ * phase .* ψk
    end

    ρ = abs.(ψ).^2
    dx = x[2] - x[1]
    N = sqrt(sum(ρ) * dx)
    return ψ ./ N
end


# Malla espacial
x = range(-30.0, 40.0, length=1200)

# Energía central del paquete (elige E0 > V0 o < V0)
E0 = 5.6         # > V0 para ver transmisión oscilatoria
k0 = sqrt(2m * E0) / ħ
σk = 0.3

# Centro inicial a la izquierda
x0 = -15.0

t0 = 0.0
ψ0 = psi_paquete(x, t0; k0=k0, σk=σk, x0=x0) 
ρ0 = abs.(ψ0).^2

#ejemplo de estacionaria quiero ver q nos jale este pedo
plot(x, ρ0,
     xlabel="x", ylabel="|ψ(x,0)|²",
     title="Paquete gaussiano incidente (t = 0)",
     legend=false)
vline!([0, a], linestyle=:dash, label="")  # para ver la barrera



# Rango de tiempos para la animación
tmax = 10.0
Nt   = 80
ts   = range(0.0, tmax, length=Nt)

ymax = 0.5   # ajusta según cómo se vea mejor

anim = @animate for (i, t) in enumerate(ts)
    ψt = psi_paquete(x, t; k0=k0, σk=σk, x0=x0)
    ρt = abs.(ψt).^2

    plot(x, ρt,
         ylim=(0, ymax), color=:purple,
         xlabel="x",
         ylabel="|ψ(x,t)|²",
         title="Paquete gaussiano (E = 7.0) con barrera de a=9.0 \n t = $(round(t, digits=2))",
         legend=false)
    plot!([0, a], [2.2, 2.2], fillrange = 0, fillalpha = 0.2,
          fillcolor = :hotpink1, linealpha = 0)
    vline!([0, a], color=:hotpink1, lw=2, alpha=0.4)
end

gif(anim, "energia_aGrange.mp4", fps=7)


## calculos para el adendo o la parte nueva esa

function transmision(k; V0=V0, a=a, m=m, ħ=ħ)
    E, regime, coeffs = scattering_coeffs(k; V0=V0, a=a, m=m, ħ=ħ)
    r, A, B, t = coeffs
    return abs2(t) #para el adendo uwu
end

function buscar_max_min_T(kmin, kmax; puntos=800) #parte dos adendo
    ks = range(kmin, kmax, length=puntos)
    Ts = [transmision(k) for k in ks]

    i_max = argmax(Ts)
    i_min = argmin(Ts)

    return ks[i_max], ks[i_min], Ts[i_max], Ts[i_min]
end


## plottear todos los casos

k_max, k_min, Tmax, Tmin = buscar_max_min_T(0.1, 5.0)


k0 = k_max
V0 = 5.0
a  = 9.0

animA = @animate for t in ts
    ψt = psi_wavepacket(x, t; k0=k0, σk=σk, x0=x0, V0=V0, a=a)
    ρt = abs.(ψt).^2
    plot(x, ρt, ylim=(0,ymax), color=:purple, lw=2,
         xlabel="x", ylabel="|ψ(x,t)|²",
         title="Paquete Gaussiano (Máximo de T), t=$(round(t,digits=2))",
         legend=false)
    plot!([0, a], [2.2, 2.2], fillrange = 0, fillalpha = 0.2,
          fillcolor = :hotpink1, linealpha = 0)
    vline!([0,a], color=:hotpink1, lw=2, alpha=0.4)
end

gif(animA, "paquete_maximo.mp4", fps=7)

k0 = k_min
V0 = 5.0
a  = 9.0

animB = @animate for t in ts
    ψt = psi_wavepacket(x, t; k0=k0, σk=σk, x0=x0, V0=V0, a=a)
    ρt = abs.(ψt).^2
    plot(x, ρt, ylim=(0,ymax),color=:purple, lw=2,
         xlabel="x", ylabel="|ψ(x,t)|²",
         title="Paquete Gaussiano (Mínimo de T), t=$(round(t,digits=2))",
         legend=false)
    plot!([0, a], [2.2, 2.2], fillrange = 0, fillalpha = 0.2,
          fillcolor = :hotpink1, linealpha = 0)
    vline!([0,a], color=:hotpink1, lw=2, alpha=0.4)
end

gif(animB, "paquete_minimo.mp4", fps=7)


k0 = k_max    # conservar mismo k0 que en el caso resonante
V0 = 0.0
a  = 0.0

#volvemos a parametrizar esto  porq sino se jode
tmax2 = 5.0
Nt   = 80
ts   = range(0.0, tmax2, length=Nt)

animC = @animate for t in ts
    ψt = psi_wavepacket(x, t; k0=k0, σk=σk, x0=x0, V0=V0, a=a)
    ρt = abs.(ψt).^2
    plot(x, ρt, ylim=(0,ymax), color=:purple, lw=2, 
         xlabel="x", ylabel="|ψ(x,t)|²",
         title="Paquete libre, t=$(round(t,digits=2))",
         legend=false)
end

gif(animC, "paquete_libre.mp4", fps=7)