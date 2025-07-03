import numpy as np
from enum import Enum

np.set_printoptions(precision=1, floatmode="fixed")

RefrigerationType = Enum("auto_refrigerado", "refrigeracao_forcada")
rho0 = 1.2  # [kg/m^3]
c0 = 343  # [m/s]
p_ref = 2e-5  # [Pa]
W_ref = 1e-12  # [W]
center_frequencies = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
# CF4 table from Barron, values in [dB] per octave bands
conversion_factors = np.array([7, 3, 9, 13, 13, 19, 24, 30])
# air absorption values from ISO 9613-1, at 20 [°C] and 50 [%] relative humidity
air_absorption_coefficients_db_per_km = np.array([2.41e-1, 7.76e-1, 2.46, 5.86, 9.14, 1.10e1, 1.32e1, 2.03e1])
air_absorption_coefficients = (air_absorption_coefficients_db_per_km * 1e-3) / (10 * np.log10(np.e))

if __name__ == "__main__":
    # room properties: surface area and volume
    S = 150  # [m^2]
    V = 300  # [m^3]
    # average absorption coefficients for the room the transformer is located in
    sabine_coefficients = np.array([0.25, 0.31, 0.48, 0.52, 0.62, 0.8, 0.8, 0.82])

    with np.printoptions(precision=2, floatmode="fixed"):
        print("----------------------------")
        print("Informações da sala:\n")
        print(f"Área da sala: {S} [m^2]")
        print(f"Volume da sala: {V} [m^3]")
        print(f"Coeficientes de absorção médios, por banda de oitava: {sabine_coefficients}")
        print("----------------------------")

    # source properties
    # insira o tipo de refrigeração
    refrigeration_type: RefrigerationType = RefrigerationType.refrigeracao_forcada
    # insira a potência aparente do transformador em [kVa]
    apparent_power = 100
    # insira a diretividade da fonte, por banda de frequência
    directivity = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    # insira a distância de medição
    r = 1  # [m]

    print("\n----------------------------")
    print("Informações da fonte e receptor:\n")
    print(f"Tipo de refrigeração: {refrigeration_type.name}")
    print(f"Potência aparente: {apparent_power} [kVa]")
    print(f"Diretividade, por banda de oitava: {directivity}")
    print(f"Distância de medição: {r} [m]")
    print("----------------------------")

    # room absorption constant, per frequency band
    R = (S * (sabine_coefficients + ((4 * air_absorption_coefficients * V) / S))) / (
        1 - sabine_coefficients - ((4 * air_absorption_coefficients * V) / S)
    )
    Lw = 45 + 12.5 * np.log10(apparent_power)
    if refrigeration_type == RefrigerationType.refrigeracao_forcada:
        Lw += 3

    Lw_oct = Lw - conversion_factors

    Lp_oct = (
        Lw_oct
        + 10 * np.log10((4 / R) + (directivity / (4 * np.pi * r**2)))
        + 10 * np.log10((rho0 * c0 * W_ref) / (p_ref**2))
    )

    print("\n----------------------------")
    print("Resultados:\n")
    print(f"Nível de potência global, sem ponderação: {Lw} [dB]")
    print(f"Nível de potência por banda de oitava, sem ponderação:       {Lw_oct} [dB]")
    print(f"Nível de pressão sonora por banda de oitava, sem ponderação: {Lp_oct} [dB]")
    print("----------------------------")
