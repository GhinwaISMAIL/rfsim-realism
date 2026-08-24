# UPV remote-driving dataset metadata request

Status: prepared, not sent  
Dataset: *Remote Driving Dataset in UPV's 5G Private network (n40)*  
DOI: https://doi.org/10.5281/zenodo.18410980  
Intended recipient: dataset authors, routed through `info@iteam.upv.es`  

## Email draft

Subject: Metadata questions about the UPV n40 remote-driving dataset (Zenodo 18410980)

Dear Dr. Lozano Teruel and colleagues,

We are using your *Remote Driving Dataset in UPV's 5G Private network (n40)*
to study measurement-equivalent calibration of an OpenAirInterface RF
simulator. We would be grateful for a few details needed to determine whether
the reported NEMO RSRP can be compared on an absolute scale with OAI's reported
SS-RSRP. We will cite the dataset and any accompanying publication in resulting
work.

1. Which Keysight NEMO version, measurement profile, and CSV export
   configuration were used?
2. Does `RSRP (NR SpCell)` represent per-SSB/per-beam RSRP, the selected best
   beam, a cell-level value, or an antenna-combined value? Was it L1- or
   L3-filtered? If so, what filter/window and reporting periodicity were used?
3. The CSVs identify the measurement type as `SSB`, while `Pathloss (NR
   SpCell)` is present but empty in all 12 UE files. If possible, could you
   share the raw NEMO files or a populated path-loss export? If not, could you
   explain how NEMO would calculate that field for this campaign?
4. What `ss-PBCH-BlockPower`, SSB EPRE, or equivalent value was signalled for
   each cell/test? Did it change between the six scenarios?
5. What were the NR carrier frequency, bandwidth, subcarrier spacing, SSB
   configuration, and serving/neighbour PCI mapping?
6. What was the conducted SSB power, separately from total carrier power?
   Please also provide the gNB antenna configuration, cable gains/losses, and
   EIRP if available.
7. Were there device-specific UE antenna-combining or calibration assumptions
   for the ASUS Snapdragon and Samsung S25 measurements?
8. The timestamps and paired GPS traces appear to place `Test_1_Asus.csv` in
   the Test 2 window and `Test_2_ASUS.csv` in the Test 1 window. Could you
   confirm whether these two ASUS filenames or scenario labels were swapped?

Configuration files, screenshots, or approximate uncertainty ranges would
also be useful if exact conducted-power details are unavailable. The purpose
of these questions is to avoid deriving a circular offset from the observed
UPV–OAI RSRP difference; we will not use that difference itself to define an
equivalence correction.

Thank you for making the dataset available.

Best regards,

[Sender name]  
[Institution or research group]  
[Reply email]

## Delivery provenance

- Zenodo lists Carles Navarro Manchón, Raúl Lozano Teruel, Iván Viciedo, Jorge
  Roche Peris, and David Gómez-Barquero as creators.
- The Zenodo record does not publish a corresponding-author email.
- The public iTEAM contact page publishes `info@iteam.upv.es`; the message
  should ask that it be forwarded to the dataset authors.
- Sending requires the sender's confirmed name, affiliation, reply address,
  and an authenticated delivery channel.
