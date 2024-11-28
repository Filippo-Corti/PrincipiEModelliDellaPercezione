N2V in pratica:
    - Grand Challenge: cosa è, perche si usa N2V in questo contesto
    - Codice Python
    - Risultati con diversi valori per epochs e batch_size
    - Difetti del N2V e possibili risoluzioni

# Noise2Void (N2V):
(
## N2V intro breve ?? no
*Noise2Void (N2V)* è una tecnica innovativa di denoising basata sul deep learning, progettata per migliorare immagini rumorose senza richiedere dati puliti per l'addestramento.
)

## Grand Challenge
N2V è stato applicato in vari contesti, tra cui il *AI4Life Microscopy Denoising Challenge*, una competizione internazionale mirata al miglioramento delle immagini di microscopia, spesso affette da rumore (strutturato o non strutturato). 

Il metodo si distingue per la capacità di rimuovere il rumore utilizzando immagini rumorose senza necessità di dati puliti per l'addestramento. Questo approccio è particolarmente utile per le applicazioni biologiche, dove ottenere immagini di riferimento prive di rumore è spesso difficile.

### Obiettivi principali
1.	*Denoising non supervisionato*: La competizione si concentra su tecniche che non richiedono immagini pulite per l'addestramento, rendendole adatte a contesti, come questo, dove raccogliere dati di riferimento è difficile o costoso.
 2.	*Applicazioni pratiche*: Migliorare la qualità delle immagini di microscopia per analisi successive come segmentazione, classificazione o estrazione di caratteristiche.

### Struttura del challenge (AGGIUNGERE)
I partecipanti applicano algoritmi su dataset biologici standardizzati.
Le performance sono valutate utilizzando metriche come PSNR (Peak Signal-to-Noise Ratio) e SSIM (Structural Similarity Index).
I metodi proposti vengono confrontati con approcci tradizionali e di nuova generazione per identificare le soluzioni migliori.

### Perché usare N2V in questo contesto?
Noise2Void (N2V) è particolarmente adatto per il contesto del challenge per le seguenti ragioni:
1. *Dati rumorosi senza immagini pulite di riferimento*:
    La microscopia biologica spesso produce immagini rumorose e ottenere dati "ground truth" privi di rumore è laborioso, se non impossibile. N2V si addestra direttamente su immagini rumorose, eliminando la necessità di dati puliti.

2.	*Generale adattabilità al rumore*:
    N2V è efficace contro vari tipi di rumore, inclusi rumori indipendenti tra pixel (tipici in dati microscopici non elaborati). Sebbene abbia limitazioni con il rumore strutturato, il suo approccio adattivo è comunque competitivo in scenari misti.

3.	*Conservazione del contenuto biologico*:
    I metodi di denoising tradizionali possono attenuare o distorcere dettagli importanti per l'analisi biologica. N2V è progettato per preservare le caratteristiche strutturali, rendendolo ideale per immagini di cellule, tessuti e altre strutture microscopiche.

## Codice Python
Le implementazioni di Noise2Void (N2V) sono facilmente accessibili attraverso librerie come *CSBDeep*, che sfrutta framework popolari come *TensorFlow* e *Keras*. Per addestrare il modello, gli script richiedono di specificare parametri essenziali come il numero di epoche, la dimensione del batch e l'architettura della rete. L'approccio standard utilizza una rete convoluzionale che integra un meccanismo per escludere alcuni pixel durante l'addestramento. Questo processo permette alla rete di sviluppare una notevole capacità di riduzione del rumore, rendendola robusta anche in presenza di dati rumorosi.

#### + aggiungere immagini di codice ??

## Risultati con valori variabili per epochs e batch_size
Un *numero maggiore di epoch* e consente al modello di convergere meglio, migliorando la sua capacità di adattarsi ai dati. Tuttavia, questo comporta il rischio di sovrallenamento, soprattutto se il modello si adatta troppo ai dettagli specifici del dataset. 
Per quanto riguarda la *dimensione del batch*, utilizzare batch più grandi può velocizzare il processo di addestramento, ma richiede una maggiore quantità di memoria. Valori comuni per il batch size sono 16 o 32. 
L’ottimizzazione di questi parametri, applicata a dataset di microscopia, ha dimostrato di migliorare significativamente metriche come il rapporto segnale-rumore (PSNR) e l’indice di somiglianza strutturale (SSIM), che misurano la qualità delle immagini denoised rispetto a quelle originali.

++ AGGIUNGERE IMMAGINI DIFFERENZA TRA EPOCHE !!


## Difetti del N2V e possibili risoluzioni (PARALRE DI N2V2 (gli altri lascia stare))
### Limitazioni
N2V può introdurre artefatti in immagini con dettagli complessi, poiché il suo approccio ignora i pixel vicini durante l'addestramento. Inoltre, non gestisce bene il rumore strutturato, che presenta correlazioni tra i pixel.

### Possibili risoluzioni
•	*Tecniche alternative*: Metodi come Noise2Self e Self2Self migliorano la robustezza utilizzando ridondanze statistiche o approcci iterativi.
•	*Dati sintetici*: Simulare rumore realistico durante l'addestramento aiuta il modello a generalizzare meglio.
•	*Architetture avanzate*: L’integrazione di Transformers o meccanismi di attenzione nelle reti convoluzionali riduce gli artefatti mantenendo elevata qualità nel denoising.

Nonostante alcune limitazioni, N2V è efficace per il denoising non supervisionato, e queste strategie ne estendono l’applicabilità in ambiti complessi come la microscopia biologica.

#### N2V2??? --> 
--> PROBLEMA N2V BLURRY IMAGES 
--> CAREamics noise to void overview https://careamics.github.io/0.1/algorithms/Noise2Void/
++ file:///C:/Users/carlo/Downloads/978-3-031-25069-9.pdf pag 525 !!!!

https://ai4life-mdc24.grand-challenge.org/
https://ai4life-mdc24.grand-challenge.org/useful-links/


# Il Rumore
Cos'è il Rumore:  
* Come si forma
* Come si classifica

