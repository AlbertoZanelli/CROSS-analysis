import ROOT
import sys
from array import array
import os
import re
import gc

ROOT.gROOT.SetBatch(True)

def extract_channel_from_filename(filename):
    base = os.path.basename(filename)
    match = re.search(r'_(-?\d+)\.root$', base)
    
    if not match:
        print(f"Warning: cannot extract channel from {base}")
        return None
    
    return int(match.group(1))

# --------------------------------------------------
# Branch buffer creator
# --------------------------------------------------
class BranchInfo:
    def __init__(self, fullName, typeName, buffer, leafDef):
        self.fullName = fullName
        self.type = typeName
        self.buffer = buffer
        self.leafDef = leafDef

def make_branch_info(prefix, name, typeName):
    full = prefix + name

    if typeName == "Int_t":
        buf = array('i', [0]); leaf = f"{full}/I"
    elif typeName == "UInt_t":
        buf = array('I', [0]); leaf = f"{full}/i"
    elif typeName == "Float_t":
        buf = array('f', [0.0]); leaf = f"{full}/F"
    elif typeName == "Double_t":
        buf = array('d', [0.0]); leaf = f"{full}/D"
    elif typeName == "Bool_t":
        buf = array('b', [0]); leaf = f"{full}/O"
    elif typeName in ("Long64_t", "Long_t"):
        buf = array('l', [0]); leaf = f"{full}/L"
    elif typeName in ("ULong64_t", "ULong_t"):
        buf = array('L', [0]); leaf = f"{full}/l"
    elif typeName == "Short_t":
        buf = array('h', [0]); leaf = f"{full}/S"
    elif typeName == "UShort_t":
        buf = array('H', [0]); leaf = f"{full}/s"
    elif typeName == "Char_t":
        buf = array('b', [0]); leaf = f"{full}/B"
    elif typeName == "UChar_t":
        buf = array('B', [0]); leaf = f"{full}/b"
    else:
        print(f"Unsupported type {typeName} for branch {full}")
        return None

    return BranchInfo(full, typeName, buf, leaf)

# --------------------------------------------------
# Copy metadata directory
# --------------------------------------------------
def copy_metadata_dir(fin, fout, newName):
    if not fin:
        return

    meta = fin.Get("module_metadata")
    if not meta:
        return

    fout.cd()
    fout.mkdir(newName)
    fout.cd(newName)

    for key in meta.GetListOfKeys():
        obj = key.ReadObj()
        if obj:
            obj.Write()
            del obj  # Explicit cleanup

# --------------------------------------------------
# Filter entries from LD1 module tree - CHUNKED VERSION
# --------------------------------------------------

def get_passing_entries(moduleTree, maxValidIndex, chunk_size=10000):
    passing = []

    if not moduleTree:
        return list(range(maxValidIndex))

    branch = moduleTree.GetBranch("isselftrigger")
    if not branch:
        print("No isselftrigger → keeping all")
        return list(range(maxValidIndex))

    val = array('b', [0])
    issign = array('b', [0])

    moduleTree.SetBranchAddress("isselftrigger", val)
    moduleTree.SetBranchAddress("issignal", issign)
    
    n = min(moduleTree.GetEntries(), maxValidIndex)
    
    # DEBUG: Check first few entries
    print(f"Total entries: {n}")
    
    # Process in chunks
    for start_idx in range(0, n, chunk_size):
        end_idx = min(start_idx + chunk_size, n)
        chunk_passing = []
        
        for i in range(start_idx, end_idx):
            moduleTree.GetEntry(i)
            if val[0] and issign[0]:  # Both must be true
                chunk_passing.append(i)
        
        passing.extend(chunk_passing)
        
        if start_idx % (chunk_size * 10) == 0:
            gc.collect()

    print(f" -> Filter: {len(passing)}/{n} pass")
    return passing

# --------------------------------------------------
# Merge trees - CHUNKED VERSION
# --------------------------------------------------
def merge_trees_chunked(treeName, t_heat, t_ld1, t_ld2, fout, passingIndices, 
                       isModule=False, chunk_size=100000):
    
    typeMaps = [{}, {}, {}]
    trees = [t_heat, t_ld1, t_ld2]
    prefixes = ["heat_", "LD1_", "LD2_"]
    allBranches = set()

    # Collect types
    for it, tree in enumerate(trees):
        if not tree:
            continue
        for b in tree.GetListOfBranches():
            leaf = b.GetLeaf(b.GetName())
            if leaf:
                name = b.GetName()
                typeMaps[it][name] = leaf.GetTypeName()
                allBranches.add(name)

    if not any(trees):
        return

    merged = ROOT.TTree(treeName, f"Merged {treeName}")
    
    # Set basket size to reduce memory usage
    merged.SetMaxTreeSize(500000000)  # 1GB max size per file
    
    branchBuffers = [[], [], []]
    isfromgoodneighbor = array('b', [0])

    # Create branches and buffers
    for name in allBranches:
        for it in range(3):
            if trees[it] is None:
                continue   
                
            typeName = typeMaps[it].get(name)
            if not typeName:
                for j in range(3):
                    if name in typeMaps[j]:
                        typeName = typeMaps[j][name]
                        break

            b = make_branch_info(prefixes[it], name, typeName)
            if not b:
                continue

            # Set smaller basket size to reduce memory usage
            branch = merged.Branch(b.fullName, b.buffer, b.leafDef)
            branch.SetBasketSize(8192)  # Smaller basket size

            if trees[it] and name in typeMaps[it]:
                if isModule and name == "isfromgoodneighbor" and it == 1:
                    trees[it].SetBranchAddress(name, isfromgoodneighbor)
                else:
                    trees[it].SetBranchAddress(name, b.buffer)

            branchBuffers[it].append(b)

    nEntries = [t.GetEntries() if t else 0 for t in trees]
    n = min([n for n in nEntries if n > 0]) if any(nEntries) else 0

    written = 0
    total_indices = len(passingIndices)

    # Process in chunks
    for chunk_start in range(0, total_indices, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_indices)
        chunk_indices = passingIndices[chunk_start:chunk_end]
        
        for idx in chunk_indices:
            if idx >= n:
                continue

            for it in range(3):
                if trees[it] and idx < nEntries[it]:
                    trees[it].GetEntry(idx)
                else:
                    for b in branchBuffers[it]:
                        b.buffer[0] = 0

            merged.Fill()
            written += 1

        # Flush to disk and force garbage collection every chunk
        merged.FlushBaskets()
        merged.AutoSave("FlushBaskets")
        gc.collect()
        
        if chunk_start % (chunk_size * 10) == 0:
            print(f"  -> Progress: {chunk_end}/{total_indices} indices processed")

    fout.cd()
    merged.Write()
    print(f"  -> wrote {written} entries")
    
    # Clean up
    del merged
    del branchBuffers
    gc.collect()

# --------------------------------------------------
# Get tree info without loading full trees
# --------------------------------------------------
def get_tree_info(files):
    """Get tree names and minimum size without keeping trees in memory"""
    allTreeNames = set()
    skipPrefixes = {
        "fouriertransform",
        "inversefouriertransform", 
        "baselinesubtraction",
        "triggerdelaycorrection",
        "global"
    }
    
    minTreeSize = float('inf')
    minTreeName = ""
    
    for f in files:
        if not f:
            continue
            
        for key in f.GetListOfKeys():
            if key.GetClassName() == "TTree":
                name = key.GetName()

                # Skip if name starts with any of the prefixes
                if any(name.startswith(prefix) for prefix in skipPrefixes):
                    continue
                    
                allTreeNames.add(name)
                
                tree = f.Get(name)
                if tree:
                    n = tree.GetEntries()
                    if n < minTreeSize:
                        if name.startswith("stabilization"):
                            continue
                        minTreeSize = n 
                        minTreeName = name
    
    return allTreeNames, minTreeSize, minTreeName

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if len(sys.argv) != 5:
    print("Usage:")
    print("python MergeProduction3.py heat.root ld1.root ld2.root merged.root")
    sys.exit(1)

heat_path = sys.argv[1]
ld1_path  = sys.argv[2]
ld2_path  = sys.argv[3]
output_path = sys.argv[4]

# Open files
heatFile = ROOT.TFile.Open(heat_path)
if not heatFile or heatFile.IsZombie():
    print("Heat file missing or corrupted → abort")
    sys.exit(1)

ld1File = None
ld2File = None

if os.path.exists(ld1_path):
    ld1File = ROOT.TFile.Open(ld1_path)
    print("LD1 found")
else:
    print("LD1 file not found → merging without LD1")

if os.path.exists(ld2_path):
    ld2File = ROOT.TFile.Open(ld2_path)
    print("LD2 found")
else:
    print("LD2 file not found → merging without LD2")

# Extract channel numbers
heatChan = extract_channel_from_filename(heat_path)
ld1Chan  = extract_channel_from_filename(ld1_path)
ld2Chan  = extract_channel_from_filename(ld2_path)

print("Detected channels:")
print("  heat =", heatChan)
print("  LD1  =", ld1Chan)
print("  LD2  =", ld2Chan)

# Get tree information efficiently
files = [heatFile, ld1File, ld2File]
allTreeNames, minTreeSize, minTreeName = get_tree_info(files)

print(f"Will process {len(allTreeNames)} trees")
print(f"Minimum tree is {minTreeName} with size: {minTreeSize}")

# Filter from heat module tree (chunked)
heatModule = heatFile.Get("module")
passingIndices = get_passing_entries(heatModule, minTreeSize, chunk_size=10000)
del heatModule  # Free memory immediately
gc.collect()

# Create output file
fout = ROOT.TFile(output_path, "RECREATE")

# Process trees one by one to minimize memory usage
for name in sorted(allTreeNames):  # Sort for consistent processing order
    print(f"Merging {name}")

    t_heat = heatFile.Get(name) if heatFile else None
    t_ld1  = ld1File.Get(name) if ld1File else None
    t_ld2  = ld2File.Get(name) if ld2File else None

    merge_trees_chunked(
        name,
        t_heat,
        t_ld1,
        t_ld2,
        fout,
        passingIndices,
        name == "module",
        chunk_size=100000  # Adjust based on available RAM
    )

    # Clean up trees immediately after processing
    del t_heat, t_ld1, t_ld2
    gc.collect()

# Create global friend tree
globalTree = ROOT.TTree("global", "Empty global tree")
fout.cd()
globalTree.Write()

for key in fout.GetListOfKeys():
    if key.GetClassName() == "TTree" and key.GetName() != "global":
        globalTree.AddFriend(key.GetName())

globalTree.Write("", ROOT.TObject.kOverwrite)
del globalTree
gc.collect()

# Copy metadata
copy_metadata_dir(heatFile, fout, "metadata_heat")
copy_metadata_dir(ld1File, fout, "metadata_LD1") 
copy_metadata_dir(ld2File, fout, "metadata_LD2")

# Add channel numbers as metadata
fout.cd()
meta = fout.mkdir("merge_metadata")
meta.cd()

ROOT.TParameter(int)("heat_channel", heatChan).Write()
ROOT.TParameter(int)("LD1_channel", ld1Chan).Write()
ROOT.TParameter(int)("LD2_channel", ld2Chan).Write()

# Close files
heatFile.Close()
if ld1File:
    ld1File.Close()
if ld2File:
    ld2File.Close()
fout.Close()

print("Done →", output_path)
print("Entries kept:", len(passingIndices))
