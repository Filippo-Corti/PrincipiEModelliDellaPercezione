# A. Argomenti per la presentazione

- Cos'è il Rumore:
    - Come si forma
    - Come si classifica

- Tecniche per la Rimozione del Rumore (Denoising):
    - Filtri Tradizionali
    - Reti Neurali
        - Cosa fanno meglio rispetto ai Filtri?
        - Due diversi casi per l'addestramento del modello:
            - Noisy + Clean Images
                - CNN
                - GAN
            - Noisy Images
                - N2N
                - N2V

- L'Algoritmo N2V
    - Caratteristiche e confronti rispetto agli altri
    - Settori dove è maggiormente utilizzato
    - Principio di Funzionamento dell'Algoritmo
        - Le Componenti Principali:
            - U-Net
            - Loss Function -> A che Grand Truth fa riferimento?
            - ...

- N2V in pratica:
    - Grand Challenge
    - Codice Python
    - Risultati con diversi valori per epochs e batch_size
    - Difetti del N2V e possibili risoluzioni

- Come l'N2V si ispira al nostro Sistema Nervoso nella Vista.

# B. Risorse

[CAREamics source code] https://github.com/CAREamics/careamics/blob/main/src/careamics/careamist.py#L423 \
[Pytorch Lightning Trainer] https://lightning.ai/docs/pytorch/stable/common/trainer.html \
[CAREamics code for the Challenge - with comparison images] https://careamics.github.io/0.1/applications/Noise2Void/JUMP/ \
[CAREamics code to test N2V on natural images] https://careamics.github.io/0.1/applications/Noise2Void/BSD68/ 

[CAREamics Noise2Void Overview] https://careamics.github.io/0.1/algorithms/Noise2Void/ \
[Original Article presenting N2V] https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf \
[ ??? ] https://proceedings.mlr.press/v97/batson19a/batson19a.pdf

[Fixing N2V Problems - Pagina 525] https://link.springer.com/chapter/10.1007/978-3-031-25069-9_33
