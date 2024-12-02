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
Noise2Void (N2V) trae ispirazione da principi biologici osservati nel nostro sistema visivo (HVS) in termini di approccio e strategia per l'elaborazione delle informazioni.

## Riempimento percettivo e mascheramento dei pixel
- HVS: il sistema visivo umano è straordinariamente adattivo e capace di compensare le lacune nei dati visivi. Un esempio lampante è il riempimento percettivo, che si verifica nella gestione dell’area cieca (blind spot): si tratta di una zona della retina in corrispondenza del punto in cui il nervo ottico si collega all'occhio. In questa zona mancano fotorecettori (coni e bastoncelli), perciò non si percepisce alcuna immagine. Nonostante ciò, le persone non sono consapevoli di questo "vuoto visivo" nella loro percezione.

    Il cervello utilizza informazioni visive provenienti dai pixel (fotorecettori) circostanti all’area cieca per "riempire" la zona mancante. In pratica, il cervello si basa su schemi, colori, texture e continuità presenti nel contesto visivo. Per esempio:
    - Se l’area cieca cade su una texture omogenea come il cielo blu, il cervello la completa riempiendola con il colore circostante.
    - Se cade su una linea retta o un contorno, il cervello estende la linea per mantenere la coerenza visiva.

- N2V: in N2V durante l'addestramento, il metodo maschera pixel casuali all'interno dell'immagine, rendendoli "invisibili" per il modello. Il valore di questi pixel viene escluso temporaneamente. Lo scopo è di addestrare il modello a predire il valore corretto di questi pixel mascherati usando solo i pixel vicini. Sfrutta la correlazione spaziale (ossia, la somiglianza tra pixel vicini) presente nelle immagini naturali. Ad esempio:
    - Se un pixel mascherato si trova in un’area di cielo azzurro uniforme, il modello predice che anche il pixel nascosto avrà un valore simile ai pixel circostanti.
    - Se il pixel si trova su un bordo o una transizione, N2V apprende a interpretare i contorni basandosi sulla configurazione dei pixel adiacenti.

    Il mascheramento fornisce un meccanismo di apprendimento auto-supervisionato. Poiché il modello non ha accesso al valore originale del pixel mascherato, è costretto a trovare schemi nei dati circostanti per fare una predizione accurata.
    In questo modo, il modello impara a distinguere il rumore (che è casuale) dai pattern sottostanti (che sono strutturati e ridondanti).

## Soppressione neuronale e mascheramento dei pixel rumorosi
- HVS: il cervello umano utilizza un meccanismo di soppressione per filtrare le informazioni visive irrilevanti o non coerenti che interferiscono con l’interpretazione corretta di un’immagine. La soppressione neurale è un meccanismo di filtraggio realizzato tramite:
    - Inibizione laterale: le cellule gangliari della retina ricevono segnali dai fotorecettori e comunicano tra loro. Questa comunicazione è regolata dall'inibizione laterale, in cui le cellule inibiscono le loro vicine per aumentare il contrasto e ridurre l’effetto del rumore.
    - Integrazione temporale: il cervello combina segnali ricevuti in momenti diversi, scartando informazioni casuali (rumore) e preservando quelle persistenti (segnale utile). Ad esempio quando osserviamo una scena poco illuminata, i segnali casuali generati dai fotorecettori sono filtrati, e il cervello costruisce un’immagine coerente basandosi su ciò che è stabile nel tempo.
    - Selezione contestuale: le aree della corteccia visiva superiore decidono quali dettagli visivi sono rilevanti e sopprimono input irrilevanti o confusi, come ombre o riflessi.

- N2V: in N2V, la stima dei pixel mascherati si basa sulla combinazione di informazioni spaziali circostanti. Quando un pixel viene mascherato, il modello si affida ai valori dei pixel vicini per stimare il valore del pixel mascherato. Questo approccio è particolarmente efficace in quanto sfrutta le correlazioni naturali che esistono tra i pixel di un'immagine.

## Adattamento dinamico al contesto e apprendimento auto-supervisionato
- HVS: l’occhio umano si adatta dinamicamente alle condizioni di illuminazione e alle caratteristiche del contesto visivo.

    Il sistema visivo umano apprende in modo dinamico attraverso l'esperienza. Non ha bisogno di avere sempre accesso a una "verità di riferimento" (un'immagine "pulita" o senza errori) per interpretare correttamente il mondo. Man mano che il cervello accumula esperienze visive, diventa sempre più abile nel riconoscere oggetti, pattern e movimenti, adattandosi al contesto circostante.

    Ad esempio se un bambino osserva una persona con gli occhiali in diverse condizioni di illuminazione, non ha bisogno di una "verità di riferimento" per riconoscere quella persona. Il cervello sviluppa la capacità di riconoscere la persona in base al contesto, anche in ambienti poco illuminati o con disturbi visivi.

- N2V: similmente, N2V affronta la riduzione del rumore senza necessitare di immagini "pulite" per l'addestramento. Il modello deve apprendere a ridurre il rumore auto-supervisionandosi, cioè sfruttando i pattern locali nelle immagini per migliorare la qualità complessiva.