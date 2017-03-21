/**
 * Created by Emmitt on 3/3/2017.
 */

//This class associate a suffix tree with its corresponding filename
public class tree_filename{
    SuffixTree tree;//Suffix tree
    String name;//Filename
    tree_filename(SuffixTree tree, String name){
        this.tree=tree;
        this.name=name;
    }
}