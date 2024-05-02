```bash
git clone ~/code/rna-ml/ nabir-public --no-local
cd nabir-public
git-filter-repo --path pairing-webapp/ --path pairclusters/
git-filter-repo --path-rename pairing-webapp:basepairs.datmos.org
git-filter-repo --path-rename pairclusters:scripts
echo "Stanislav Lukeš <github@exyi.cz> <exyi@outlook.cz>" > mailmap
echo "Stanislav Lukeš <github@exyi.cz> <git@exyi.cz>" >> mailmap
echo "Stanislav Lukeš <github@exyi.cz> <github@exyi.cz>" >> mailmap
git-filter-repo --mailmap ./mailmap
rm mailmap
echo "regex:(Stanislav |Standa )?(Lukeš)? [<](exyi@outlook.cz|git@exyi.cz|github@exyi.cz)[>]==>Stanislav Lukeš <github@exyi.cz>" > replace_text
git-filter-repo --replace-text ./replace_text
rm replace_text
```
