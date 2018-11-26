## MAP-Elites for Constrained Optimization

#### TODO Notes

- Creare un algoritmo più generale possibile
	- Utente deve poter inserire in modo dinamico i constraints e le funzioni obiettivo
	- Come anche i limit dei constraints e la definizione dei bins
- Nel caso i constraints siano più di due si dovrebbe cercare di avere comunque una visualizzazione. Per esempio nel caso si abbiamo 3 o 4 dimensioni si può avere una visualizzazione dei quelle dim all'interno di ogni singolo quadretto della mappa 2d.
- La cella in basso a sinistra rappresenta il caso in cui tutti i constraints sono soddisfatti (no errore). In pratica gli assi della mappa rappresentano quanto ci discostiamo dal valore corretto.
- Sarebbe carino creare degli slider per far cambiare in modo veloce il valore dei bin delle varie dimensioni (dipende dai tempi di esecuzione dell'algoritmo.


## Points to reflect and improve on

- Initial population generation. We can either generate the initial population in the genotype space (with no control over what bins in the features space will be filled) or in the feature space. In the second case we should find some way of doing that.
- Should we think about supporting other *solutions* data types (apart from a tuple of float)

## TODO

- [ ] Make heatmap plotting available for > 2 dimensions
- [ ] Add config file (logging.ini) based logging
- [ ] Manage inf values in Bokeh