import numpy as np

def afficher_tableau(tableau, idx_base, etape):
    """
    Affiche le tableau du simplexe à chaque étape
    """
    print(f"\n=== Étape {etape} ===")
    m, n = tableau.shape
    n -= 1  # exclure la dernière colonne (b)
    print("Tableau du simplexe :")
    print(np.round(tableau, 3))
    print("Variables de base :", ["x" + str(i+1) for i in idx_base])
    print("-" * 40)


def simplex(c, A, b, afficher=True):
    """
    Méthode du Simplexe - Maximisation :
    Max Z = c^T x
    sous contraintes A x <= b et x >= 0
    """

    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    m, n = A.shape  # m contraintes, n variables

    # Construction du tableau initial
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = c  # coefficients de la fonction objectif

    idx_base = np.arange(n, n + m)  # indices des variables de base (écart)
    etape = 0

    if afficher:
        afficher_tableau(tableau, idx_base, etape)

    while True:
        ligne_obj = tableau[-1, :-1]

        # Condition d'arrêt : plus de coefficient positif
        if np.all(ligne_obj <= 1e-9):
            break

        # Sélection de la colonne pivot
        col_pivot = np.argmax(ligne_obj)
        col = tableau[:m, col_pivot]
        rhs = tableau[:m, -1]

        # Calcul des rapports b_i / a_i,j (uniquement si a_i,j > 0)
        ratios = np.full(m, np.inf)
        positif = col > 1e-9
        ratios[positif] = rhs[positif] / col[positif]

        if np.all(np.isinf(ratios)):
            print("⚠️ Le problème est non borné.")
            return None, None

        # Ligne pivot = plus petit ratio positif
        ligne_pivot = np.argmin(ratios)
        pivot = tableau[ligne_pivot, col_pivot]

        # Normalisation
        tableau[ligne_pivot] /= pivot

        # Élimination des autres lignes
        for i in range(m + 1):
            if i != ligne_pivot:
                coeff = tableau[i, col_pivot]
                tableau[i] -= coeff * tableau[ligne_pivot]

        # Mise à jour de la base
        idx_base[ligne_pivot] = col_pivot
        etape += 1

        if afficher:
            print(f"\nPivot choisi : ligne {ligne_pivot+1}, colonne {col_pivot+1}")
            afficher_tableau(tableau, idx_base, etape)

    # Extraction de la solution
    x = np.zeros(n)
    for i in range(m):
        if idx_base[i] < n:
            x[idx_base[i]] = tableau[i, -1]

    z = tableau[-1, -1]

    print("\n✅ Solution optimale trouvée !")
    print(f"x = {np.round(x, 3)}")
    print(f"Z = {np.round(z, 3)}")
    return x, z


if __name__ == "__main__":
    # Exemple : problème de maximisation
    # Max Z = 50x1 + 60x2
    # sous contraintes :
    # x1 + 2x2 <= 8
    # 2x1 + 2x2 <= 10
    # 9x1 + 4x2 <= 36
    A = np.array([
        [1, 2],
        [2, 2],
        [9, 4]
    ])
    b = np.array([8, 10, 36])
    c = np.array([50, 60])

    simplex(c, A, b, afficher=True)
