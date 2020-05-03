# We call it MUSKET, the Muenster Skeleton Tool for High-Performance Code Generation.

## Purpose
Musket is a high-level approach to simplify the creation of parallel programs. It is based on the domain-specific-language Musket which uses the concepts of algorithmic skeletons. 

## Usage
### Requirements:
1. An eclipse installation and java
### Installation
1. Help > Install new software...
    - “Add...“ new update site (Name: Xtext; Location: http://download.eclipse.org/modeling/tmf/xtext/updates/composite/releases/)
    - Install “XtendIDE” and “XtextComplete SDK” from the Xtextgroup (currently version 2.13)
2. Restart Eclipse
3. Import 5 musket projects de.wwu.musket{~, ~.ide, ~.tests., ~.ui, ~.ui.tests} into workspace
4. Right-click on de.wwu.musket/src/GenerateMusket.mwe2 > Run as > MWE2 workflow 
5. When asked “Do you agree to download ...”, type y in the console of Eclipse and hit enter
6. To start the editor start a new Eclipse instance (“Launch Runtime Eclipse” from the Run dropdown menu).
7. If this doesn’t exist, go to Run dropdown menu > “Run configurations” > Add new “Eclipse Application”
8. In the new (inner) Eclipse, import project de.wwu.musket.models
