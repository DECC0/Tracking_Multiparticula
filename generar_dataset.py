"""
Punto de entrada del simulador (paquete simulador_psf): genera un dataset de
videos sinteticos de fluorescencia con emisores en movimiento browniano 3D.

Imprime el resumen del montaje, la compacidad de la PSF vs z y la SNR pico,
y escribe en el directorio de trabajo:
    videos/video_XXX.tif  (T, H, W) uint16, un archivo por video
    ground_truth.csv      video_id, frame, particle_id, x_px, y_px, z_nm,
                          photons, D_um2_s, en_fov

Uso (reemplaza a "python sim_imagenesPSF.py", con las mismas salidas):
    ~/miniforge3/envs/tesis/bin/python generar_dataset.py
"""

from dataclasses import replace

from simulador_psf import (Config, BancoPSF, generar_dataset, guardar_gt,
                           energia_encerrada, imprimir_snr_pico)


def main():
    cfg = Config(mascara="esferica")    # <- "esferica" o "ninguna"
    print(cfg.resumen())

    banco = BancoPSF(cfg)
    print(f"\nmascara: {cfg.mascara}")
    print(f"energia fuera del recorte de {cfg.K_psf} px: "
          f"max {100*banco.energia_recortada.max():.1f}%")

    p0 = banco.psf_en_z(0).max()
    zs = [-800, -600, -400, -200, 0, 200, 400, 600, 800]
    print("\ncompacidad de la PSF vs z:")
    print("   z [nm]   pico rel.   energia r<=2px")
    for z in zs:
        print(f"  {z:6.0f}   {banco.psf_en_z(z).max()/p0:8.3f}   "
              f"{energia_encerrada(banco, z):12.3f}")

    # SNR pico por z, y el mismo calculo sin aberracion (Phi = 0) para comparar
    imprimir_snr_pico(banco)
    if cfg.mascara != "ninguna":
        imprimir_snr_pico(BancoPSF(replace(cfg, mascara="ninguna")))

    gt = generar_dataset(cfg, banco)
    guardar_gt(gt, "ground_truth.csv")
    print(f"\nvideos/          : {cfg.n_videos} videos de "
          f"({cfg.n_frames}, {cfg.H}, {cfg.W}) uint16")
    print(f"ground_truth.csv : {gt.particle_id.groupby(gt.video_id).nunique().sum()} "
          f"trayectorias, {gt.en_fov.mean()*100:.0f}% de puntos dentro del FOV")


if __name__ == "__main__":
    main()
