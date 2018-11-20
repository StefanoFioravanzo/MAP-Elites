## MAP-Elites for Constrained Optimization

#### TODO Notes

- Creare un algoritmo più generale possibile
	- Utente deve poter inserire in modo dinamico i constraints e le funzioni obiettivo
	- Come anche i limit dei constraints e la definizione dei bins
- Nel caso i constraints siano più di due si dovrebbe cercare di avere comunque una visualizzazione. Per esempio nel caso si abbiamo 3 o 4 dimensioni si può avere una visualizzazione dei quelle dim all'interno di ogni singolo quadretto della mappa 2d.
- La cella in basso a sinistra rappresenta il caso in cui tutti i constraints sono soddisfatti (no errore). In pratica gli assi della mappa rappresentano quanto ci discostiamo dal valore corretto.
- Sarebbe carino creare degli slider per far cambiare in modo veloce il valore dei bin delle varie dimensioni (dipende dai tempi di esecuzione dell'algoritmo.


#### Questions for professor:

- Come generare la popolazione iniziale. Se faccio una generazione random nello spazio degli input avrei una probabilita altissima di riempire solamente il bin 0,0 oppure quelli che vanno all'infinito
- Quindi per generare la pop iniziale piuttosto sto nello spazio delle features
- Al momento l'argoritmo assume uno spazio di input (quindi delle soluzioni) di semplici tuple di float. Quindi sto creando un array n-dimnesionale dove ogni cella contiene una tupla di float (la soluzione). La cosa forse più sarebbe quella di creare un oggetto (dataclass) che tenga la rappresentazione di una soluzione
