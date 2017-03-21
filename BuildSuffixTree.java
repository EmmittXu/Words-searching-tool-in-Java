import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Emmitt on 3/3/2017.
 */
public class BuildSuffixTree {
    //For every file in dir_path, the buildTree function creates a suffix tree and
    // corresponds the tree to the file it belongs to
    public static List<tree_filename> buildTree(String dir_path) throws IOException {
        String content="";
        List<tree_filename> treeList=new ArrayList<tree_filename>();
        //Get all the file names in the dir_path
        ReadFile files=new ReadFile();
        List<String> nameList=files.getAllFileName(dir_path);
        if(nameList.size()==0) return treeList;

        //Get the contents of each file and build this file's suffix tree
        for(String name:nameList){
            content=files.readFile(dir_path+name);
            String []words=content.split(" ");
            SuffixTree sTree=new SuffixTree(words[0]);
            List<SuffixTree.Node> links=new ArrayList<SuffixTree.Node>();
            for(String word:words)
                if(!word.equals("")) {
                    links = sTree.createSuffixTree(word, sTree.root, links);
            }
            treeList.add(new tree_filename(sTree,name));
        }
        //The returned value is a list that contains all suffix trees and their
        //corresponding file names
        return treeList;
    }
}
