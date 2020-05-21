def k_means(im): 
    
    '''
    param im - tablica numpy

    Kompresja obrazu za pomocą grupowania k-średnich.
    a) Każdy piksel obrazu RGB zawiera po jednej wartości 
    dla każdego z 3 kanałów: R (red), G (green) i B (blue). 
    Korzystając z grupowania metodą k-średnich, możemy 
    podzielić wszystkie piksele danego obrazu na k klastrów 
    i każdemu pikselowi przypisać kolor reprezentowany 
    przez najbliższego centroida (tj. współrzędne środka 
    ciężkości, do którego dany piksel został przydzielony). 
    Dzięki temu działaniu obraz składający się z milionów 
    kolorów może zostać skompresowany do obrazu 
    zawierającego jedynie k kolorów.'''

    '''Moduł ma zostać stworzony tak, że główna funkcja 
    sterująca przyjmie na starcie TYLKO zaimportowany 
    wcześniej obraz 'landscape.jpg' (przekonwertowany 
    wcześniej w module main do tablicy numpy) oraz ma 
    zwracać JEDYNIE trójwymiarową tablicę numpy z 
    nowymi wartościami pikseli.'''

    # zewnętrzne biblioteki
    import skimage.io as io
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    #print(im.shape)
    io.imshow(im)

    '''
    # Podział na kanały rgb.
    red = im[:, :, 0] # czyli obraz o oryginalnych wymiarach x, y, ale tylko pierwszy kanał kolorystyczny - czerwień
    green = im[:, :, 1] # czyli obraz o oryginalnych wymiarach x, y, ale tylko drugi kanał kolorystyczny - zieleń
    blue = im[:, :, 2] # czyli obraz o oryginalnych wymiarach x, y, ale tylko trzeci kanał kolorystyczny - błękit

    # Tak graficznie przedstawia się każdy kanał z osobna
    
    fig, axs = plt.subplots(2,2)

    cax_00 = axs[0,0].imshow(im)
    axs[0,0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
    axs[0,0].yaxis.set_major_formatter(plt.NullFormatter())  # kill ylabels

    cax_01 = axs[0,1].imshow(red, cmap='Reds')
    fig.colorbar(cax_01, ax=axs[0,1])
    axs[0,1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0,1].yaxis.set_major_formatter(plt.NullFormatter())

    cax_10 = axs[1,0].imshow(green, cmap='Greens')
    fig.colorbar(cax_10, ax=axs[1,0])
    axs[1,0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1,0].yaxis.set_major_formatter(plt.NullFormatter())

    cax_11 = axs[1,1].imshow(blue, cmap='Blues')
    fig.colorbar(cax_11, ax=axs[1,1])
    axs[1,1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1,1].yaxis.set_major_formatter(plt.NullFormatter())
    plt.show()

    # Plot histograms
    fig, axs = plt.subplots(3, sharex=True, sharey=True)

    axs[0].hist(red.ravel(), bins=10) # histogram z podziałem na dziesięć "słupków"
    axs[0].set_title('Red')
    axs[1].hist(green.ravel(), bins=10)
    axs[1].set_title('Green')
    axs[2].hist(blue.ravel(), bins=10)
    axs[2].set_title('Blue')

    plt.show() 
    '''
    # implementacja k-means wymusza na nas przekonwertowanie trójwymiarowej tablicy na dwuwymiarową.
    w, h, d = tuple(im.shape) # rozpakowanie
    assert d == 3 # sprawdzenie czy dobrze podzielone
    imar = np.reshape(im, (w*h, d)) # "spłaszczenie" płaszczyzny wymiarów obrazu do ciągu. Równoznaczne z przyznaniem każdemu pikselowi niepowtarzalnego ID i przyporządkowaniem mu koloru.

    '''
    b) Dokonać klasteryzacji metodą k-średnich i wybrać do
     ostatecznego podziału jako liczbę k najmniejszą z liczb, 
     dla której wartość inercji jest mniejsza niż 1.4*10^8. 
     
     UWAGA! 
     
     Grupowania dokonywać dla argumentu random_state=1, tzn. 
     funkcja KMeans ma przyjąć jako wartość argumentu 
     random_state liczbę 1, zaś początkowe miejsca centroidów 
     mają zostać wybrane metodą 'k-means++'. Wszystkie inne 
     rozwiązania będą odrzucane.
    '''
    '''
    def k_means_clustering(data, n_clusters, init, epochs):
        clustering = KMeans(n_clusters=n_clusters, random_state=1,
                            init=init, max_iter=epochs).fit(data)
        # Sprawdźmy, gdzie zostały przypisane dane punkty
        assignment = clustering.labels_
        # Zobaczmy, jak wyglądają środki ciężkości grup
        centroids = clustering.cluster_centers_
        # Dokonajmy wizualizacji podziału danych
        fig, ax = plt.subplots()
        points = ax.scatter(data[:, 0], data[:, 1], c=assignment)
        # Środki ciężkości klastrów zaznaczymy na czerwono
        centres = ax.scatter(centroids[:, 0], centroids[:, 1], c='red')
        legend = ax.legend(*points.legend_elements(),
                        title='klaster')
        ax.add_artist(legend)
        plt.show()
        
        # Zwrócimy model, przyporządkowanie dla naszego zbioru oraz centroidy
        return clustering, assignment, centroids

    make_cluster, assigment, centroids = k_means_clustering(imar, 6, 'k-means++', 100)
    '''
    inertia_measures = []
    arguments = []
    centroids = []
    i = 8
    limit = 1.4*(10**8)
    # Nauczmy model dla różnej liczby klastrów i policzmy dla każdej wersji wartość inercji.
    while i < 21:
        centroids = []
        clustering = KMeans(n_clusters=i, init='k-means++',
                            max_iter=100, random_state=1)
        clustering.fit(imar)
        inertia_measures.append(clustering.inertia_)
        centroids = clustering.cluster_centers_
        
        arguments.append(i)
        if inertia_measures[-1] < limit:
            break
        i+=1 
    
    '''    
    # przybliżenie wykresu
    fig, ax = plt.subplots()
    ax.plot(arguments[-5:], inertia_measures[-5:], marker='o')
    ax.set_xlabel('Liczba klastrów')
    ax.set_ylabel('Wartość inercji')
    plt.show()
    f'Najmniejsza liczba klastrów dla której inercja jest mniejsza niż 1.4*10^8 to {arguments[-1]}. Dla tej wartości inercja jest równa {inertia_measures[-1]}
    '''

    # c) Każdą współrzędną wszystkich uzyskanych centroidów zaokrąglić do najbliższej liczby całkowitej.
    centroids = np.round(centroids)
    # print(centroids)

    # d) Dla grupowania na tak ustalone k klastrów dokonać przypisania każdemu pikselowi jego zaokrąglonej wartości środka ciężkości.
    centr = centroids[:]
    for en, pixel in enumerate(imar):
        j = 0 # index na którym jest centroid najbliższy pikselowi
        dist = np.linalg.norm(pixel-centr[0]) # odległość euklidesowa zainicjowana pierwszą odległością dla każdego piksela do pierwszego centroidu
        for idx, val in enumerate(centr):
            nd = np.linalg.norm(pixel-val) # nowa odległość
            if nd < dist:
                dist, j = nd, idx # szukanie najmniejszej odległości rgb od centroidu
        imar[en] = centr[j]

    # e) Nowo utworzony obraz zapisać w pliku .png i dołączyć go do rozwiązania.
    # odtworzenie obrazka w formie skompresowanej
    compr = np.zeros([w, h, 3], dtype=np.uint8)
    pixel = 0
    for j in range(w):
        for i in range(h):
            compr[j, i] = imar[pixel]
            pixel += 1
    #print(compr.shape)
    #io.imshow(compr)
    io.imsave("compressed.png", compr)
    return compr


if __name__ == "__main__":
    import skimage.io as io
    import numpy as np
    import matplotlib.pyplot as plt
    # Wczytanie obrazu - skimage.io automatycznie przechowuje obrazy jako numpy array.
    landscape = io.imread('landscape.jpg')
    nowa_tablica = k_means(landscape)
    io.imsave("compressed_main.png", nowa_tablica)



