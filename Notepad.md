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

[CAREamics source code] https://github.com/CAREamics/careamics/blob/main/src/careamics/careamist.py#L423 

[Pytorch Lightning Trainer] https://lightning.ai/docs/pytorch/stable/common/trainer.html 

[CAREamics code for the Challenge - with comparison images] https://careamics.github.io/0.1/applications/Noise2Void/JUMP/ 

[CAREamics code to test N2V on natural images] https://careamics.github.io/0.1/applications/Noise2Void/BSD68/ 

[CAREamics Noise2Void Overview] https://careamics.github.io/0.1/algorithms/Noise2Void/ 

[Original Article presenting N2V] https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf 

[ ??? ] https://proceedings.mlr.press/v97/batson19a/batson19a.pdf

[Fixing N2V Problems - Pagina 525] https://link.springer.com/chapter/10.1007/978-3-031-25069-9_33

[More on N2V] https://www.toolify.ai/ai-news/denoising-images-with-noise2void-a-deep-learning-approach-1150583

[N2V Code Implementation] https://github.com/juglab/n2v

[N2V2 - Contains details for N2V] https://www.researchgate.net/publication/365448865_N2V2_--_Fixing_Noise2Void_Checkerboard_Artifacts_with_Modified_Sampling_Strategies_and_a_Tweaked_Network_Architecture

[An example experiment with N2V with Architecture details] https://pmc.ncbi.nlm.nih.gov/articles/PMC8563445/pdf/nihms-1749463.pdf

[Another example of N2V with params] https://colab.research.google.com/github/r3gm/InsightSolver-Colab/blob/main/N2V_for_Image_Denoising_of_Single_Channel_Images__Training_and_Inference.ipynb

[Architecture Template] https://github.com/BiaPyX/BiaPy/blob/master/templates/denoising/2d_denoising.yaml

[Perplexity.ai answer to which params are important for N2V] https://www.perplexity.ai/search/i-have-zero-knowledge-of-cnns-R9QNnIGPSeGw.uHlJkE9yg#4

[U-Net explained] https://arxiv.org/pdf/1505.04597v1

[Youtube Video(s)] https://www.youtube.com/watch?v=nVKvGBq_-wQ