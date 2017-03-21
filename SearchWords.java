import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.io.File;

/**
 * Created by Emmitt on 3/3/2017.
 */

//User interface to enter commands
public class SearchWords {
    public static void main(String args[]) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String accStr = "";
        System.out.println("Welcome to words searching!");
        System.out.println("This program allows you to search words, sentences or substrings in a directory specified by user");
        System.out.println("Return name of the file that conatains the targets");
        System.out.println("For commands please type: help");

        while (true) {
            accStr = br.readLine();
            if(accStr.equals("quit"))break;
            if(accStr.equals(""))continue;
            String[]command=accStr.split(" ");
            if(accStr.equals("help")) {
                System.out.println("***************************");
                System.out.println("To search words, type:");
                System.out.println("find <<words> ..... > <path-to-directory>");
                System.out.println("You should eliminate white space within the substring manually");
                System.out.println("Example: 'find NiceToMeetYou Ilovecoding Java data'   will find");
                System.out.println("3 substrings 'NiceToMeetYou', 'Ilovecoding' and 'Java' in all text files under folder data");
                System.out.println("You can always specify <path-to-directory> as data, which contains several sample files");
                System.out.println();
                System.out.println("To list all file information, type:");
                System.out.println("ls-all <path-to-directory>");
                System.out.println();
                System.out.println("To exit the program, type:");
                System.out.println("quit");
                System.out.println("***************************");
            }
            else if(command[0].equals("find")){
                System.out.println("Searching.........");
                long start = System.currentTimeMillis();
				String sep= File.separator;
                List<tree_filename> TreeList = BuildSuffixTree.buildTree(command[command.length-1]+sep);
                if(TreeList.size()==0)continue;
                long end = System.currentTimeMillis();
                System.out.println((float)(end-start)/1000+"s");
                List<String> targets=new ArrayList<String>();
                for(int i=1; i<command.length-1;i++){
                    targets.add(command[i]);
                }
                for(String target:targets){
                    System.out.println("Searching word(s): "+target);
                    boolean exist=false;
                    for (tree_filename item : TreeList) {
                        if (item.tree.search(target)){
                            exist=true;
                            System.out.println("Found it in file: " + item.name);
                        }
                    }
                    if(exist==false)
                        System.out.println("No such words in directory "+command[command.length-1]);
                    System.out.println("**********************");
                }
                System.out.println("**********************");
            }
            else if(command[0].equals("ls-all")&&command.length==2){
                ReadFile files =new ReadFile();
                List<String> TreeList = files.getAllFileName(command[1]);
                System.out.println("This directory contains "+TreeList.size()+" files");
                for(String file:TreeList){
                    System.out.println(file);
                }
                System.out.println("**********************");
            }
            else{
                System.out.println("Command unknown! Type <help> for more information");
                System.out.println("**********************");
            }
        }
        System.out.print("Program runs over");
    }
}
