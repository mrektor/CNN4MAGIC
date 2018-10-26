# void Extract_pixel_info(
# ){
#
#     TChain mc("Events");
# mc.Add("/nfs/magicsim/dani/sorcerer/M1/GA_M1_za05to35_0_10*");
#
# mc.SetBranchStatus("*" ,0); / /All branches disabled
# mc.SetBranchStatus("MCerPhotEvt*" ,1);
#
# MCerPhotEvt *cerphot = new
# MCerPhotEvt;
# mc.SetBranchAddress("MCerPhotEvt.", & cerphot);
#
# ofstream
# fout;
# fout.open("Data_sample.txt");
#
# for (Int_t i = 0; i < mc.GetEntries();
# i + +) // We
# fill
# histograms
# {
#     mc.GetEvent(i);
#
# for (int j=0; j < cerphot->GetNumPixels();
# j + +){
#     MCerPhotPix & pix = (*cerphot)[j];
# Nph = pix.GetNumPhotons();
# std::cout << "Event " << i << " Pixel " << j << " Nph " << Nph << std::endl;
# if (i == 2)
# {fout << Nph << endl;}
# }
# }
# fout.close();
# }
