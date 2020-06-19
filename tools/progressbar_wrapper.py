from __future__ import print_function
import time

import tools.progressbar.progressbar as progressbar
import tools.progressbar.widgets as progressbar_widgets


class ProgressBar(object):
    def __init__(self):
        self.TotalResults = []
        self.NumberOfFinishedResults = 0

    def Update(self):
        self.ProgressBar.update(self.NumberOfFinishedResults)
        return

    def CallbackForProgressBar(self, res=''):
        """
        Callback function for pool.async if the progress bar needs to be displayed.
        Must use with DisplayProgressBar function.

        :param multiprocessing.pool.AsyncResult res: Result got from callback function in pool.async.
        """
        self.NumberOfFinishedResults += 1
        self.TotalResults.append(res)
        return

    def DisplayProgressBar(self, ProcessingResults, ExpectedResultsSize, CheckInterval=1, type="minute"):
        '''
        Display a progress bar for multiprocessing. This function should be used after pool.close(No need to use pool.join anymore).
        The call back function for pool.async should be set as CallbackForProgressBar.

        :param multiprocessing.pool.AsyncResult ProcessingResults: Processing results returned by pool.async.
        :param int ExpectedResultsSize: How many result you will reveive, i.e. the total length of progress bar.
        :param float CheckInterval: How many seconds will the progress bar be updated. When it's too large, the _main program may hang there.
        :param String type: Three types: "minute", "hour", "second"; corresponds displaying iters/minute iters/hour and iters/second.
        '''
        self.ProcessingResults = ProcessingResults
        ProgressBarWidgets = [progressbar_widgets.Percentage(),
                              ' ', progressbar_widgets.Bar(),
                              ' ', progressbar_widgets.SimpleProgress(),
                              ' ', progressbar_widgets.Timer(),
                              ' ', progressbar_widgets.AdaptiveETA()]
        self.ProgressBar = progressbar.ProgressBar(ExpectedResultsSize, ProgressBarWidgets)
        self.StartTime = time.time()
        PreviousNumberOfResults = 0
        self.ProgressBar.start()
        while self.ProcessingResults.ready() == False:
            self.Update()
            time.sleep(CheckInterval)
        time.sleep(CheckInterval)
        self.Update()
        self.ProgressBar.finish()
        self.EndTime = time.time()
        print("Processing finished.")
        # print "Processing results: ", self.TotalResults
        print("Time Elapsed: %.2fs, or %.2fmins, or %.2fhours" % (
            (self.EndTime - self.StartTime), (self.EndTime - self.StartTime) / 60,
            (self.EndTime - self.StartTime) / 3600))
        print("Processing finished.")
        print("Processing results: " + str(self.TotalResults))
        print("Time Elapsed: %.2fs, or %.2fmins, or %.2fhours" % (
        (self.EndTime - self.StartTime), (self.EndTime - self.StartTime) / 60, (self.EndTime - self.StartTime) / 3600))
        return