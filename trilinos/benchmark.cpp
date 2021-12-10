#include <Tpetra_Core.hpp>
#include <Tpetra_Version.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include <Xpetra_Matrix.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_IO.hpp>

#include <iostream>
#include <string>
#include <chrono>

typedef int32_t OrdinalType;
typedef double ScalarType;

bool endsWith(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

Teuchos::RCP<Teuchos::FancyOStream> getOutputStream(const Teuchos::Comm<int>& comm) {
  using Teuchos::getFancyOStream;
  const int myRank = comm.getRank();
  if (myRank == 0) {
    // Process 0 of the given communicator prints to std::cout.
    return getFancyOStream(Teuchos::rcpFromRef(std::cout));
  }
  else {
    // A "black hole output stream" ignores all output directed to it.
    return getFancyOStream(Teuchos::rcp(new Teuchos::oblackholestream()));
  }
}

void spmv(Teuchos::RCP<const Teuchos::Comm<int>> comm, Teuchos::RCP<Tpetra::CrsMatrix<double>> A, int warmup, int niter, Teuchos::FancyOStream& out) {
  // Create the x and y vectors.
  Tpetra::Vector<double> x(A->getDomainMap());
  Tpetra::Vector<double> y(A->getRangeMap()); 
  // Fill the vectors with values.
  x.putScalar(1.0);
  y.putScalar(0.0);

  // Warmup iterations.
  for (int i = 0; i < warmup; i++) {
    A->apply(x, y);
  }
  // Timed iterations.
  auto timer = Teuchos::TimeMonitor::getNewCounter("SpMV");
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niter; i++) {
    Teuchos::TimeMonitor timeMon(*timer);
    A->apply(x, y);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto avg = double(ms.count()) / double(niter);
  Teuchos::TimeMonitor::summarize(comm.ptr(), out);
  out << "Average time: " << avg << " ms." << std::endl;
}

int main(int argc, char** argv) {

  Tpetra::ScopeGuard tpetraScope (&argc, &argv);
  Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm ();

  // Get an output stream that only process 0 can communicate on.
  Teuchos::RCP<Teuchos::FancyOStream> pOut = getOutputStream(*comm);
  Teuchos::FancyOStream& out = *pOut;
  
  Teuchos::CommandLineProcessor cmdp(false, true);
  std::string filename = "";
  cmdp.setOption("file", &filename, "Path and filename of the matrix to be read.");

  // TODO (rohany): Experiment with different data distributions.
  std::string distribution ="1D";
  cmdp.setOption("distribution", &distribution, "Parallel distribution to use: 1D, 2D, LowerTriangularBlock, MMFile");

  int chunkSize = 1000;
  cmdp.setOption("loadChunkSize", &chunkSize, "Number of edges to be read and broadcasted at once");

  int warmup = 10;
  cmdp.setOption("warmup", &warmup, "Number of warmup iterations to run");

  int niter = 20;
  cmdp.setOption("n", &niter, "Number of timed iterations to run");

  if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    std::cout << "Command line parse unsuccessful" << std::endl;
    return 1;
  }
  
  if (filename.empty()) {
    std::cout << "Specify filename with --file=<filename>" << std::endl;
    return 1;
  }

  auto isMtxMarket = endsWith(filename, ".mtx");
  auto xpetraMat = Xpetra::IO<double, int, long long>::Read(filename, Xpetra::UseTpetra, comm, !isMtxMarket);
  auto matrixWrap = Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrixWrap<double, int, long long>>(xpetraMat);
  auto xPetraTpetraMat = Teuchos::rcp_dynamic_cast<Xpetra::TpetraCrsMatrix<double, int, long long>>(matrixWrap->getCrsMatrix());
  auto A = xPetraTpetraMat->getTpetra_CrsMatrixNonConst();
  // Teuchos::FancyOStream foo(Teuchos::rcp(&std::cout, false));
  // A->describe(foo, Teuchos::VERB_EXTREME);

  spmv(comm, A, warmup, niter, out);

  return 0;
}
