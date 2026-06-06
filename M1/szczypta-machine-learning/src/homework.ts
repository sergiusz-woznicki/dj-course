import { addMatrices, multiplyMatrices, transpose, assertMatricesDimensionMatch, assertMatricesCompatible } from "./matrix-operations";
import { fromJSONFile, jsonFilePath, randomizeMatrix, randomizeVector } from "./utils";
import { vectorSum, dotProduct } from "./vector-operations";
import { Matrix, Vector } from "./types";
import { displayVector, displayMatrix } from "./display";

// HINT: (w zaleności od wybranego kierunku implementacji) może być mnożenie macierzy przez wektory - tę operację będzie trzeba zaimplementować 😉 
// ale nie jest to konieczne 😎

// HINT: w mnożeniu macierzy kolejność ma znaczenie - bo w zależności od kolejności albo wymiary obydwu składników pasują do siebie albo nie.

// HINT: wstań od komputera i przemyśl problem. Serio. Zastanów się, ile linijek wystarczy aby podać rozwiązanie :)
// (traktując "linijkę" jako pojedynczą operację na tensorach) 😎

// PROŚBA: jeśli znasz rozwiązanie, to nie spamuj discorda - a przynajmniej nie od razu. Pozwól innym pomóżdżyć 😎

const cases = ['case-1.json', 'case-2.json', 'case-3.json', 'case-4.json'];

for (const caseFile of cases) {
  const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath(caseFile));

  console.log(`\n═══════════════════════════════════════`);
  console.log(`  ${caseFile}`);
  console.log(`═══════════════════════════════════════`);

  console.log('WK_Matrix');
  console.log(displayMatrix(WK_Matrix, -1));
  console.log('WQ_Matrix');
  console.log(displayMatrix(WQ_Matrix, -1));
  console.log('X_Input_Matrix');
  console.log(displayMatrix(X_Input_Matrix, -1));

  // Attention score matrix S = Q · K^T
  // Q = X · WQ, K = X · WK

  const Q_Matrix = multiplyMatrices(X_Input_Matrix, WQ_Matrix);
  const K_Matrix = multiplyMatrices(X_Input_Matrix, WK_Matrix);
  const K_Transposed = transpose(K_Matrix);
  const S_Matrix = multiplyMatrices(Q_Matrix, K_Transposed);

  console.log('Q_Matrix (X · WQ)');
  console.log(displayMatrix(Q_Matrix, -1));
  console.log('K_Matrix (X · WK)');
  console.log(displayMatrix(K_Matrix, -1));
  console.log('K_Transposed');
  console.log(displayMatrix(K_Transposed, -1));
  console.log('S_Matrix (Q · K^T)');
  console.log(displayMatrix(S_Matrix, -1));
}
