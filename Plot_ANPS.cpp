#include <TFile.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TApplication.h>
#include <TAxis.h> 
#include <TH1D.h>
#include <iostream>

int main(int argc, char** argv) {
    TApplication app("app", &argc, argv);

    // ... (Apertura file e recupero grafici identici a prima) ...
    TFile *file1 = TFile::Open("RUN89/Processed/Processed_20251210T182259_000089_25.root", "READ");
    if (!file1 || file1->IsZombie()) { std::cerr << "Error opening file1" << std::endl; return 1; }
    TFile *file2 = TFile::Open("RUN143/Processed/Processed_20260302T091555_000143_25.root", "READ");
    if (!file2 || file2->IsZombie()) { std::cerr << "Error opening file2" << std::endl; return 1; }
    
    TGraph *graph1 = dynamic_cast<TGraph*>(file1->Get("averagepowerspectrum_anps_normalized;1"));
    if (!graph1) { std::cerr << "Errore: graph1 non e' un TGraph!" << std::endl; return 1; }
    
    TGraph *graph2 = dynamic_cast<TGraph*>(file2->Get("averagepowerspectrum_anps_normalized;1"));
    if (!graph2) { std::cerr << "Errore: graph2 non e' un TGraph!" << std::endl; return 1; }

    TCanvas *c1 = new TCanvas("c1", "Analisi ANPS", 800, 600);
    c1->SetLogx();
    c1->SetLogy();
    c1->SetGrid(); 

    graph1->SetLineColor(kBlue);
    graph1->SetLineWidth(2);
    graph2->SetLineColor(kRed);
    graph2->SetLineWidth(2);

    // 3. Modifica le opzioni di Draw. 
    // Usare "A" per il primo TGraph per disegnare gli assi, e "L" per disegnare come linea continua
    graph1->Draw("AL"); 
    graph2->Draw("L");

    // 4. Configurazione assi (questo ora funzionerà perfettamente!)
    graph1->GetXaxis()->SetTitle("Frequency [Hz]");
    graph1->GetYaxis()->SetTitle("Power Spectrum [V^{2}/Hz]"); 
    graph1->GetXaxis()->CenterTitle();
    graph1->GetYaxis()->CenterTitle();

    // 5. I limiti logaritmici
    graph1->SetMinimum(1e-15); 
    graph1->SetMaximum(1e-5);

    // --- 4. LEGENDA ---
    TLegend *leg = new TLegend(0.65, 0.75, 0.88, 0.88);
    leg->AddEntry(graph1, "December 2025", "l");
    leg->AddEntry(graph2, "March 2026", "l");
    leg->SetBorderSize(1);
    leg->Draw();

    c1->Update();
    std::cout << "Plot pronto in scala Log-Log." << std::endl;
    
    app.Run();
    return 0;
}