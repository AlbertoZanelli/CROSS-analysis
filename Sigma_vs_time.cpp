#include <iostream>
#include <vector>
#include <string>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TApplication.h>
#include <TStyle.h>
#include <TGraph.h>
#include <TF1.h>
#include <TSystemDirectory.h>
#include <TSystemFile.h>
#include <TList.h>

using namespace std;

int main(int argc, char** argv) {
    TApplication app("app", &argc, argv);

    // Evita che ROOT tenga in memoria gli istogrammi dopo la chiusura del file
    TH1::AddDirectory(kFALSE);

    // --- 1. DEFINIZIONE CARTELLA ---
    const char* dirPath = "./dati/"; // INSERISCI QUI IL PERCORSO DELLA TUA CARTELLA
    TSystemDirectory dir(dirPath, dirPath);
    TList *files = dir.GetListOfFiles();

    if (!files) {
        cerr << "Errore: Cartella " << dirPath << " non trovata o vuota." << endl;
        return 1;
    }

    files->Sort(); // Ordina i file alfabeticamente

    // Vettori per memorizzare i risultati
    vector<double> sigmas;
    vector<double> file_indices;
    int file_count = 0;

    cout << "Inizio analisi dei file nella cartella: " << dirPath << endl;

    // --- 2. CICLO SU TUTTI I FILE NELLA CARTELLA ---
    for (int i = 0; i < files->GetSize(); ++i) {
        TSystemFile *fileEntry = (TSystemFile*)files->At(i);
        string fname = fileEntry->GetName();

        // Considera solo i file con estensione .root
        if (!fileEntry->IsDirectory() && fname.find(".root") != string::npos) {
            string fullPath = string(dirPath) + "/" + fname;
            TFile *file = TFile::Open(fullPath.c_str(), "READ");
            
            if (!file || file->IsZombie()) {
                cerr << "Impossibile aprire il file " << fname << ", salto..." << endl;
                continue;
            }

            cout << "Elaborazione file: " << fname << "..." << flush;

            // --- 3. RECUPERO ALBERI E COLLEGAMENTO FRIENDS ---
            TTree *tree_stab = (TTree*)file->Get("stabilization_all;2");
            TTree *tree_opt  = (TTree*)file->Get("optimumfilter_all;2");
            TTree *tree_cal  = (TTree*)file->Get("calibration_rough;2");

            if (!tree_stab || !tree_opt || !tree_cal) {
                cerr << " Alberi mancanti, salto file." << endl;
                file->Close();
                continue;
            }

            tree_stab->AddFriend(tree_opt);
            tree_stab->AddFriend(tree_cal);

            // --- 4. VARIABILI E BRANCH ADDRESS ---
            double heat_amp = 0, cal_rough = 0, corr = 0;
            bool bad_heat = false, heat_is = false;
            int heat_trig = 0;

            tree_stab->SetBranchAddress("heat_amplitude", &heat_amp);
            tree_stab->SetBranchAddress("heat_issignal", &heat_is);
            tree_stab->SetBranchAddress("heat_numberoftriggers", &heat_trig);
            tree_stab->SetBranchAddress("heat_correlation", &corr);
            tree_stab->SetBranchAddress("heat_badinterval", &bad_heat);
            tree_cal->SetBranchAddress("heat_amplitude", &cal_rough);

            // --- 5. ISTOGRAMMA E CICLO SUGLI EVENTI ---
            // Usa un range adeguato per il tuo picco da fittare
            TH1F *h_param = new TH1F("h_param", "Parametro da fittare", 50, 7440, 7650);

            Long64_t nEntries = tree_stab->GetEntries();
            for (Long64_t j = 0; j < nEntries; j++) {
                tree_stab->GetEntry(j);

                // Applicazione dei tagli (modifica a seconda del parametro che vuoi plottare)
                if (!bad_heat && heat_trig == 1 && heat_is && corr > 0.9999 && cal_rough > 2550 && cal_rough < 2610) {
                    h_param->Fill(heat_amp);
                }
            }

            // --- 6. FIT GAUSSIANO ED ESTRAZIONE SIGMA ---
            if (h_param->GetEntries() > 10) { // Assicurati di avere abbastanza eventi per il fit
                h_param->Fit("gaus", "Q"); // "Q" = Quiet mode (non stampa troppi dettagli a schermo)
                TF1 *fitGaus = h_param->GetFunction("gaus");
                
                if (fitGaus) {
                    double sigma = fitGaus->GetParameter(2); // Parametro 0: Costante, 1: Media, 2: Sigma
                    sigmas.push_back(sigma);
                    file_indices.push_back(file_count);
                    file_count++;
                    cout << " Sigma = " << sigma << endl;
                } else {
                    cout << " Fit fallito." << endl;
                }
            } else {
                cout << " Pochi eventi, fit ignorato." << endl;
            }

            // --- 7. PULIZIA MEMORIA ---
            delete h_param;
            file->Close();
        }
    }

    // --- 8. VISUALIZZAZIONE RISULTATI ---
    if (sigmas.empty()) {
        cerr << "Nessun fit eseguito con successo." << endl;
        return 1;
    }

    gStyle->SetOptStat(1111);

    TCanvas *c_res = new TCanvas("c_res", "Andamento Sigma", 800, 600);
    c_res->SetGrid();

    TGraph *g_sigmas = new TGraph(sigmas.size(), &file_indices[0], &sigmas[0]);
    g_sigmas->SetTitle("Andamento della Risoluzione (#sigma); Indice File; #sigma (Amplitude)");
    g_sigmas->SetMarkerStyle(20);
    g_sigmas->SetMarkerSize(1.0);
    g_sigmas->SetMarkerColor(kBlue);
    g_sigmas->SetLineColor(kBlue);

    g_sigmas->Draw("APL"); // Disegna assi (A), punti (P) e linee tra i punti (L)

    cout << "Analisi completata. Grafico pronto." << endl;
    
    app.Run();
    return 0;
}