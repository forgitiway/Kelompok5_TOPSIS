from mpi4py import MPI
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = MPI.Wtime()

weights = np.array([0.3, 0.4, 0.3])

criteria = np.array([-1, 1, 1])

if rank == 0:

    df = pd.read_csv("dataset.csv", nrows=400000)

    df.columns = df.columns.str.strip()

    data = df[
        [
            'brand_name',
            'price',
            'average_rating',
            'total_reviews'
        ]
    ].copy()

    data['price'] = pd.to_numeric(
        data['price'],
        errors='coerce'
    ).fillna(0)

    data['average_rating'] = pd.to_numeric(
        data['average_rating'],
        errors='coerce'
    ).fillna(0)

    data['total_reviews'] = pd.to_numeric(
        data['total_reviews'],
        errors='coerce'
    ).fillna(0)

    data['brand_name'] = data['brand_name'].fillna("Unknown")

    total_data = len(data)

    X = data[
        [
            'price',
            'average_rating',
            'total_reviews'
        ]
    ].values.astype(float)

    pembagi = np.sqrt((X**2).sum(axis=0))

    pembagi[pembagi == 0] = 1

    norm = X / pembagi

    weighted = norm * weights

    ideal_pos = np.zeros(weighted.shape[1])
    ideal_neg = np.zeros(weighted.shape[1])

    for j in range(len(criteria)):

        if criteria[j] == 1:
            # BENEFIT
            ideal_pos[j] = weighted[:, j].max()
            ideal_neg[j] = weighted[:, j].min()

        else:
            # COST
            ideal_pos[j] = weighted[:, j].min()
            ideal_neg[j] = weighted[:, j].max()

    chunks = np.array_split(weighted, size)

else:
    data = None
    chunks = None
    ideal_pos = None
    ideal_neg = None
    total_data = None

ideal_pos = comm.bcast(ideal_pos, root=0)
ideal_neg = comm.bcast(ideal_neg, root=0)

local_weighted = comm.scatter(chunks, root=0)

Dp_local = np.sqrt(
    ((local_weighted - ideal_pos) ** 2).sum(axis=1)
)

Dn_local = np.sqrt(
    ((local_weighted - ideal_neg) ** 2).sum(axis=1)
)

V_local = Dn_local / (Dp_local + Dn_local)

all_scores = comm.gather(V_local, root=0)

if rank == 0:

    V = np.concatenate(all_scores)

    hasil = data.copy()

    hasil['score'] = V

    brand_summary = (
        hasil.groupby('brand_name')
        .agg(
            jumlah_produk=('score', 'count'),
            rata_rata_score=('score', 'mean')
        )
        .sort_values(
            by='rata_rata_score',
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    brand_summary['rank'] = (
        brand_summary.index + 1
    )

    brand_summary['rata_rata_score'] = (
        brand_summary['rata_rata_score']
        .round(4)
    )

    print("\n" + "=" * 70)
    print("TOP 10 BRAND TERBAIK (TOPSIS MPI)")
    print("=" * 70)

    print(
        brand_summary[
            [
                'rank',
                'brand_name',
                'jumlah_produk',
                'rata_rata_score'
            ]
        ].to_string(index=False)
    )

    end_time = MPI.Wtime()

    print("\n" + "=" * 70)
    print(f"Total data diproses : {total_data:,}")
    print(f"Jumlah proses MPI   : {size}")
    print(f"Waktu eksekusi      : {end_time - start_time:.4f} detik")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PEMBAGIAN DATA SETIAP PROSES")
    print("=" * 70)

    for i, chunk in enumerate(chunks):
        print(f"Rank {i} memproses {len(chunk):,} data")

comm.Barrier()