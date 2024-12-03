# TECNICHE PER LA RIMOZIONE DEL RUMORE

## Denoising
L'obiettivo del denoising è ridurre il rumore nell'immagine per ottenere una stima che approssimi il più possibile l'immagine originale minimizzando il rumore senza compromettere le caratteristiche essenziali dell'immagine. Questo comporta diverse sfide:
- Le aree piatte devono essere lisce: zone con colori o intensità uniformi non devono mostrare variazioni spurie introdotte dal rumore.
- I bordi devono essere preservati: i contorni e i dettagli significativi non devono essere sfocati durante il processo di denoising.
- Le texture devono essere mantenute: pattern fini o regolari non devono essere erroneamente interpretati come rumore.
- Non devono essere generati nuovi artefatti: il metodo di denoising non deve introdurre deformazioni visibili o strutture artificiali.

## Tecniche di denoising

### 1. Spatial domain filtering
I metodi di denoising nel dominio spaziale cercano di rimuovere il rumore direttamente sui pixel dell'immagine, sfruttando la correlazione tra i pixel o i gruppi di pixel vicini.

Il filtraggio è uno strumento centrale nell'elaborazione delle immagini, ed è utilizzato per ridurre il rumore modificando i valori di intensità dei pixel. I filtri nel dominio spaziale possono essere classificati in due categorie principali:
- **Filtri lineari**: calcolano la nuova intensità di un pixel come combinazione lineare delle intensità dei pixel vicini
    - Filtro medio (Mean Filter): sostituisce il valore di un pixel con la media dei valori dei pixel circostanti
    - Filtro gaussiano (Gaussian Filter): calcola il valore di un pixel tramite una funzione gaussiana che assegna maggior peso ai pixel centrali nella finestra.
- **Filtri non lineari**: utilizzano funzioni non lineari per calcolare i nuovi valori di intensità, permettendo di preservare meglio i dettagli rispetto ai filtri lineari
    - Filtro mediano (Median Filter): sostituisce il valore di un pixel con la mediana dei valori dei pixel circostanti. Efficace per rimuovere il rumore impulsivo (sale e pepe).
    - Filtro bilaterale (Bilateral Filter): Combina informazioni spaziali e di intensità. Sostituisce il valore di un pixel con una media pesata dei valori circostanti, dove i pesi dipendono dalla distanza spaziale e dalla differenza di intensità. Preserva i bordi mentre riduce il rumore.

I filtri spaziali agiscono come filtri passa-basso, attenuando le componenti ad alta frequenza (associate al rumore) e preservando quelle a bassa frequenza (associate alle regioni uniformi dell'immagine), tuttavia l'eliminazione delle componenti ad alta frequenza può anche causare perdita di dettagli e bordi.

Filtrare il rumore senza sfumare i bordi e le texture è una sfida. Filtri lineari tendono a sfumare l'intera immagine, mentre i filtri non lineari, come il filtro bilaterale, cercano di mitigare questo effetto. 

I filtri semplici, come il filtro medio o mediano, sono efficienti e facili da implementare, ma i filtri più avanzati, come il filtro bilaterale, richiedono maggior potenza computazionale, specialmente con immagini di grandi dimensioni.

### 2. Transform domain filtering
I metodi di denoising basati sulle trasformazioni rappresentano un'evoluzione rispetto ai metodi nel dominio spaziale, offrendo una maggiore capacità di separare il rumore dai dettagli dell'immagine. Questi metodi sfruttano l'osservazione che le caratteristiche dell'immagine e del rumore si distinguono meglio in un dominio trasformato.

L'immagine rumorosa viene trasformata in un dominio alternativo utilizzando una specifica funzione di trasformazione. 

Una volta trasformata l'immagine, si applica un filtro per ridurre il rumore. Questo filtraggio si basa sulle differenze nelle proprietà dei coefficienti associati a contenuto utile dell'immagine e rumore.

Dopo il filtraggio, l'immagine viene riconvertita dal dominio trasformato al dominio spaziale utilizzando la trasformazione inversa. Questo passaggio ricombina i coefficienti filtrati per generare un'immagine ripulita. La qualità dell'immagine ricostruita dipende dalla precisione con cui il filtraggio è stato effettuato: 
- Se il filtro è troppo aggressivo, potrebbero verificarsi perdite di dettagli importanti.
- Se il filtro è troppo conservativo, potrebbe rimanere una quantità significativa di rumore.

Questi metodi si differenziano in base alla funzione di trasformazione utilizzata. Le principali categorie includono:
- **Trasformazioni non adattive** (non-data-adaptive): si basano su trasformazioni standardizzate e su modelli generici che separano le caratteristiche dell'immagine dal rumore, senza tener conto delle peculiarità specifiche dell'immagine in analisi. Le tecniche più comuni sono:
    - Trasformata di Fourier: converte l'immagine dal dominio spaziale al dominio della frequenza. Una volta trasformata, il denoising avviene tramite filtri passabasso che riducono i componenti di alta frequenza (associati al rumore), mantenendo le basse frequenze (associate alle caratteristiche principali dell'immagine).
    - Trasformata wavelet: scompone l'immagine in una rappresentazione multi-scala, che consente di separare i dettagli a diverse risoluzioni. 
- **Trasformazioni adattive** (data-adaptive): utilizzano tecniche che si adattano alle caratteristiche specifiche dell'immagine e del rumore. Le trasformazioni sono calcolate direttamente dai dati dell'immagine, rendendo il metodo più flessibile rispetto ai metodi non adattivi. Questi metodi includono strumenti come Analisi delle Componenti Indipendenti (ICA) e Analisi delle Componenti Principali (PCA), che cercano di separare le componenti del segnale utili da quelle corrotte dal rumore. 

### 3. Metodi avanzati
Uno dei metodi più popolari per il denoising delle immagini è BM3D (Block-Matching and 3D Filtering).

BM3D si basa su una combinazione di tecniche di:
- **Matching di blocchi**: l'immagine viene suddivisa in piccoli blocchi sovrapposti. Per ogni blocco di riferimento, si cercano blocchi simili in una regione circostante dell'immagine. I blocchi simili vengono raggruppati in un "cluster" tridimensionale.
- **Filtraggio 3D**: i gruppi 3D sono trasformati nel dominio wavelet e viene applicato un filtro su questi gruppi. 
- **Ricostruzione dell'immagine**: dopo il filtraggio, i blocchi ripuliti vengono trasformati nuovamente nello spazio immagine.

BM3D è tipicamente implementato in due fasi principali:
1. *Filtraggio di base* (Basic Estimation): si esegue il matching di blocchi e il filtraggio 3D iniziale. Questo passaggio produce una stima preliminare dell'immagine pulita.
2. *Filtraggio raffinato* (Refinement Stage): si ripete il matching di blocchi e il filtraggio 3D, ma questa volta si utilizza l'immagine preliminare come riferimento. Questo passaggio migliora ulteriormente i dettagli fini.

### 4. Metodi basati sull'apprendimento automatico
I principali metodi basati sull’apprendimento automatico sono:
- **Convolutional neural network** (CNN): funzionano applicando filtri alle immagini in modo iterativo per identificare e rimuovere il rumore.

    Un'architettura famosa è DnCNN (Denoising CNN). DnCNN impara a predire il rumore presente nell'immagine invece di ricostruire direttamente l'immagine pulita. Una volta ottenuto il rumore, lo si sottrae dall'immagine originale per ottenere un'immagine denoised: IMG PULITA = IMG RUMOROSA - RUMORE PREDETTO. In questo modo più facile per la rete predire il rumore rispetto a ricostruire l'immagine pulita direttamente. Questo implica una convergenza più rapida del modello in fase di addestramento e un output più preciso.

    Per addestrare il modello, si utilizza un set di dati di immagini pulite (senza rumore) e si aggiunge rumore sintetico per creare immagini rumorose.

    Processo di addestramento:
    - Input: l'immagine rumorosa.
    - Output previsto: il rumore sintetico aggiunto.
    - Funzione di perdita: si calcola la differenza tra il rumore predetto dalla rete e il rumore vero (quello aggiunto).
    - Ottimizzazione: l'algoritmo di backpropagation ottimizza i pesi della rete per ridurre la perdita

    La sfida principale in questo contesto è che se il modello è addestrato solo su un tipo specifico di rumore, può non funzionare bene su altri tipi.

- **Autoencoder**: è composto da due reti neurali principali:
    - Encoder: comprime l'input in una rappresentazione più compatta chiamata codifica latente, con l’obiettivo di ridurre la dimensione dei dati, catturandone solo le caratteristiche essenziali. 
    - Decoder: prende questa codifica latente e tenta di ricostruire il dato originale. 

    Quando si utilizza un autoencoder per rimuovere il rumore da un'immagine in input viene fornita al modello una versione rumorosa dell'immagine. La rete viene addestrata per fare in modo che il suo output (immagine ricostruita) assomigli il più possibile all'immagine pulita originale. Il modello apprende a ignorare il rumore durante la fase di codifica. Alla fine, l'autoencoder produce una versione denoised dell'immagine.

- **Generative Adversarial Networks** (GAN): è composta da due reti neurali che lavorano in competizione:
    - Generatore (Generator): nel caso del denoising a partire da un'immagine rumorosa cerca di produrre una versione pulita dell'immagine.
    - Discriminatore (Discriminator): valuta l'immagine generata dal Generatore confrontandola con immagini reali (senza rumore). Se il Discriminatore rileva differenze, segnala al Generatore di migliorare il suo output.

    La competizione tra queste reti permette al Generatore di migliorarsi continuamente, producendo immagini sempre più realistiche.

I metodi appena descritti richiedono coppie di immagini (una rumorosa e una pulita). Tuttavia esistono contesti in cui raccogliere immagini pulite è difficile o impossibile, ad esempio, in ambito medico o astronomico.

Due approcci innovativi e particolarmente efficienti per affrontare questo problema sono:
- **Noise2Noise** (N2N) è un approccio di machine learning utilizzato per la rimozione del rumore da immagini o segnali, senza la necessità di avere un esempio pulito (senza rumore) da cui apprendere. L'idea centrale di Noise2Noise è che, anziché allenare un modello a partire da coppie di immagini contenenti un'immagine rumorosa e una pulita (come accade nei tradizionali metodi di denoising), si può allenare un modello utilizzando solo due immagini rumorose che sono entrambe versioni contaminate dello stesso dato di base. 

    Vengono forniti al modello due immagini, entrambe affette da rumore, ma provenienti dalla stessa scena o dal medesimo oggetto. Queste immagini sono "rumorose" in modi diversi, ma essendo basate sullo stesso contenuto, il modello può imparare a recuperare la versione "pulita" sottostante, anche senza avere mai un esempio perfetto (senza rumore).

- **Noise2Void** (N2V) è una tecnica di denoising in immagini, che si basa su un approccio simile al Noise2Noise ma con una caratteristica distintiva. A differenza di Noise2Noise, che utilizza due immagini rumorose per l'addestramento, Noise2Void può essere applicato anche a un solo dato rumoroso, senza la necessità di immagini aggiuntive. L'idea di Noise2Void è che si può addestrare una rete neurale a rimuovere il rumore da una singola immagine rumorosa senza bisogno di una versione pulita di riferimento.

    Durante l'addestramento, l'algoritmo maschera casualmente alcuni dei pixel dell'immagine rumorosa, e la rete neurale cerca di predire il valore di un pixel mascherato utilizzando i pixel vicini non mascherati.

# RUMORE NEL SISTEMA VISIVO UMANO
Nel sistema visivo umano, il termine "rumore" si riferisce a fonti di disturbo che influenzano la capacità di percepire e interpretare correttamente gli stimoli visivi. Il rumore può essere suddiviso in due categorie principali: rumore esterno e rumore interno.
- **Rumore Esterno**: questo tipo di rumore proviene dall'ambiente circostante e include fattori come scarsa illuminazione, bagliori, riflessi o la presenza di altri oggetti nel campo visivo, che possono compromettere la chiarezza delle informazioni visive.
- **Rumore Interno**: questo rumore ha origine da fattori neurali e biologici all'interno del sistema visivo. Esso è causato principalmente dalle fluttuazioni casuali nell'attività neuronale, che possono verificarsi anche in assenza di stimoli visivi esterni. Le principali cause di rumore interno includono:
    - Attività Spontanea dei Neuroni: anche in assenza di stimoli visivi, i neuroni visivi (localizzati nella retina, nel nervo ottico e nella corteccia visiva) continuano a generare impulsi elettrici. Questa attività spontanea introduce variabilità nei segnali elaborati, interferendo con la chiarezza delle informazioni visive.
    - Variazioni nella Trasmissione Sinaptica: la comunicazione tra neuroni avviene tramite sinapsi, e in questo processo possono esserci fluttuazioni nei livelli di neurotrasmettitori rilasciati e nella sensibilità dei recettori post-sinaptici. Queste fluttuazioni introducono incertezza nella trasmissione dei segnali visivi, particolarmente quando gli stimoli esterni sono deboli.
    - Interazioni Casualizzate tra Reti Neurali: interazioni non correlate o casuali tra diverse aree neurali del sistema visivo possono generare ulteriori fluttuazioni, aumentando il rumore e complicando l’elaborazione delle informazioni visive.
 

# SOMIGLIANZE N2V E SISTEMA VISIVO UMANO
Il funzionamento di Noise2Void (N2V), un algoritmo per la rimozione del rumore nelle immagini, presenta alcune somiglianze con i principi di base del sistema visivo umano (HVS). Questi parallelismi emergono in tre ambiti principali: utilizzo del contesto, apprendimento e gestione delle illusioni.

## Utilizzo del contesto
- N2V: l'algoritmo utilizza il contesto locale di un'immagine per ricostruire il valore di un pixel rumoroso. Durante l’addestramento, alcuni pixel vengono mascherati, rendendoli "invisibili" al modello, che deve predire il loro valore basandosi solo sui pixel circostanti. Questo approccio si basa su un principio fondamentale delle immagini naturali: i pixel vicini tendono a essere altamente correlati, poiché le strutture visive (come bordi o gradienti) sono generalmente continue e prevedibili.

- HVS: allo stesso modo, il sistema visivo umano utilizza il contesto per compensare eventuali mancanze nella percezione visiva. Un esempio significativo è il riempimento percettivo che si verifica nella gestione del punto cieco (blind spot), una regione della retina priva di fotorecettori. Nonostante l'assenza di informazioni visive in questa zona, il cervello "riempie" la lacuna sfruttando:
    - le informazioni provenienti dall'altro occhio, se disponibile.
    - le informazioni contestuali fornite dai pixel circostanti, come linee continue, gradienti e texture.

    Un altro esempio, quando osserviamo un oggetto parzialmente occluso, il cervello utilizza il contesto locale e le conoscenze pregresse sugli oggetti per ricostruire una rappresentazione visiva coerente.

## Apprendimento
- N2V: è un modello di apprendimento supervisionato che si addestra usando solo immagini rumorose, senza bisogno di un'immagine pulita come riferimento. Durante il processo di apprendimento, il modello identifica e sfrutta le relazioni statistiche nel dataset rumoroso per prevedere il segnale originale.

- HVS: anche il sistema visivo umano apprende dinamicamente attraverso l’esperienza, senza necessitare di una "verità di riferimento" (ad esempio, un’immagine perfettamente chiara). Il cervello migliora progressivamente le proprie capacità percettive man mano che accumula esperienze visive, diventando sempre più abile nel riconoscere pattern, oggetti e movimenti.

    Apprendimento percettivo: Si tratta del miglioramento delle prestazioni in compiti sensoriali attraverso la pratica ripetuta. Questo tipo di apprendimento non riduce il rumore interno (causato dalle fluttuazioni neurali), ma aumenta la capacità del cervello di estrarre e utilizzare il segnale rilevante. Una teoria chiave è che il cervello confronta gli stimoli visivi con una serie di modelli memorizzati per identificare correttamente ciò che percepisce.


## Illusioni
- N2V: una limitazione di N2V è la difficoltà nel distinguere tra rumore strutturato (caratterizzato da schemi ripetuti o correlati) e il segnale reale. Se il rumore presenta caratteristiche simili al segnale, come un pattern a righe, l'algoritmo potrebbe interpretarlo erroneamente come parte del segnale, compromettendo il risultato finale.

- HVS: Analogamente, il sistema visivo umano può essere ingannato da illusioni ottiche che sfruttano schemi ambigui o rumore strutturato. Le illusioni dimostrano come il cervello, nel tentativo di interpretare il mondo, può cadere in errore quando il contesto o le informazioni disponibili portano a una rappresentazione visiva errata.